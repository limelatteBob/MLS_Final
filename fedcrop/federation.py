"""Real Flower SecAgg+ over a serialized in-process message transport.

This is a single-host protocol simulator, not process isolation or a production
authenticated network. ClientApp/secaggplus_mod perform the actual cryptography.
"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, RecordDict, ConfigRecord, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.serde import message_to_proto, message_from_proto
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage
from flwr.compat.common import recorddict_compat as compat
from flwr.server import LegacyContext
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD, MAIN_CONFIGS_RECORD
from flwr.supercore.task_identity import TaskIdentity


class StaleUpdateError(RuntimeError):
    pass


def validate_version(requested, current):
    if requested != current:
        raise StaleUpdateError("Stale model version")


class LocalGrid:
    def __init__(self, clients, app, *, drop_ids=(), drop_stage=Stage.COLLECT_MASKED_VECTORS):
        self.contexts = {i + 1: Context(run_id=1, node_id=i + 1, node_config={"client_id": i},
                                      state=RecordDict(), run_config={}) for i in clients}
        self.app = app
        self.drop_ids = {i + 1 for i in drop_ids}
        self.drop_stage = drop_stage
        self.bytes_by_stage = defaultdict(int)
        self.bytes_by_node = defaultdict(int)
        self.survivors = []

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            node = message.metadata.dst_node_id
            stage = message.content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]
            encoded = message_to_proto(message)
            size = encoded.ByteSize()
            # Serialize/deserialize so a client cannot mutate the server message object.
            message = message_from_proto(encoded)
            reply = None
            if not (node in self.drop_ids and stage == self.drop_stage and stage != Stage.COLLECT_MASKED_VECTORS):
                TaskIdentity.node_id = node
                TaskIdentity.task_id = node
                try:
                    reply = self.app(message, self.contexts[node])
                finally:
                    TaskIdentity.node_id = 0
                    TaskIdentity.task_id = 0
                # A collection-stage dropout occurs AFTER local work; its privacy is spent.
                if node in self.drop_ids and stage == self.drop_stage:
                    reply = None
            if reply is not None:
                if stage == Stage.COLLECT_MASKED_VECTORS and any(len(record) for record in reply.content.array_records.values()):
                    raise RuntimeError("Refusing a SecAgg reply containing unmasked parameter arrays")
                encoded_reply = message_to_proto(reply)
                size += encoded_reply.ByteSize()
                replies.append(message_from_proto(encoded_reply))
            self.bytes_by_node[node - 1] += size
            self.bytes_by_stage[stage] += size
        if messages and stage == Stage.COLLECT_MASKED_VECTORS:
            self.survivors = [m.metadata.src_node_id - 1 for m in replies]
        return replies


class RoundStrategy(FedAvg):
    def __init__(self, selected, version):
        super().__init__(min_fit_clients=2, min_available_clients=2, accept_failures=True)
        self.selected = selected
        self.version = version
        self.result = None

    def configure_fit(self, server_round, parameters, client_manager):
        return [(SimpleNamespace(node_id=i + 1), FitIns(parameters, {"version": self.version})) for i in self.selected]

    def aggregate_fit(self, server_round, results, failures):
        # SecAgg+ replaces each result with the SAME recovered aggregate.
        # Do not independently re-weight or average the masked payloads.
        if results:
            self.result = parameters_to_ndarrays(results[0][1].parameters)
            return results[0][1].parameters, {}
        return None, {}


class GuardedSecAggPlusWorkflow(SecAggPlusWorkflow):
    """Fail closed on early key-exchange dropout in the pinned Flower version.

    Flower 1.43 retains early missing peers in sampled_node_ids, then requests
    shares which were never distributed. Abort BEFORE training in that case.
    Collection/unmask dropouts retain the upstream threshold recovery protocol.
    """
    def setup_stage(self, grid, context, state):
        ok = super().setup_stage(grid, context, state)
        return ok and state.active_node_ids == state.sampled_node_ids

    def share_keys_stage(self, grid, context, state):
        ok = super().share_keys_stage(grid, context, state)
        return ok and state.active_node_ids == state.sampled_node_ids


def secure_round(weights, selected, train_fn, version, *, drop_ids=(),
                 drop_stage=Stage.COLLECT_MASKED_VECTORS, threshold=4):
    if len(selected) < threshold:
        return None, {"survivors": [], "wire_bytes": 0, "bytes_by_node": {}, "bytes_by_stage": {}}

    class Client(NumPyClient):
        def __init__(self, client_id):
            self.client_id = client_id

        def fit(self, parameters, config):
            validate_version(config["version"], version)
            result = train_fn(self.client_id, parameters)
            if any(not np.isfinite(x).all() for x in result):
                raise ValueError("Non-finite local update")
            # Fixed-point encoding clipping is part of the protocol and is documented.
            return result, 1, {}  # Equal farm weights; no private dataset sizes/metrics.

    app = ClientApp(client_fn=lambda context: Client(context.node_config["client_id"]).to_client(),
                    mods=[secaggplus_mod])
    grid = LocalGrid(selected, app, drop_ids=drop_ids, drop_stage=drop_stage)
    strategy = RoundStrategy(selected, version)
    state = RecordDict()
    state.config_records[MAIN_CONFIGS_RECORD] = ConfigRecord({"current_round": version + 1})
    state.array_records[MAIN_PARAMS_RECORD] = compat.parameters_to_arrayrecord(ndarrays_to_parameters(weights), True)
    context = LegacyContext(context=Context(run_id=1, node_id=0, node_config={}, state=state, run_config={}),
                            strategy=strategy)
    flow = GuardedSecAggPlusWorkflow(num_shares=len(selected), reconstruction_threshold=threshold,
                             max_weight=1.0, clipping_range=8.0)
    previous = (TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id)
    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 1, 0, 0
    try:
        flow(grid, context)
    finally:
        TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id = previous
    return strategy.result, {"survivors": grid.survivors, "wire_bytes": sum(grid.bytes_by_stage.values()),
                             "bytes_by_stage": dict(grid.bytes_by_stage), "bytes_by_node": dict(grid.bytes_by_node)}


def plain_round(weights, selected, train_fn, *, drop_ids=(), threshold=4):
    results, survivors, byte_count, by_node = [], [], 0, {}
    download = sum(len(x) for x in ndarrays_to_parameters(weights).tensors)
    for client in selected:
        local = train_fn(client, weights)
        size = download
        if client not in drop_ids:
            results.append(local)
            survivors.append(client)
            size += sum(len(x) for x in ndarrays_to_parameters(local).tensors)
        by_node[client] = size
        byte_count += size
    aggregate = [np.mean(np.stack(x), axis=0) for x in zip(*results)] if len(results) >= threshold else None
    return aggregate, {"survivors": survivors, "wire_bytes": byte_count, "bytes_by_node": by_node, "bytes_by_stage": {}}
