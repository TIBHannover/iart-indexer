# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import indexer_pb2 as indexer__pb2


class IndexerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.list_plugins = channel.unary_unary(
            "/iart.indexer.Indexer/list_plugins",
            request_serializer=indexer__pb2.ListPluginsRequest.SerializeToString,
            response_deserializer=indexer__pb2.ListPluginsReply.FromString,
        )
        self.status = channel.unary_unary(
            "/iart.indexer.Indexer/status",
            request_serializer=indexer__pb2.StatusRequest.SerializeToString,
            response_deserializer=indexer__pb2.StatusReply.FromString,
        )
        self.build_suggester = channel.unary_unary(
            "/iart.indexer.Indexer/build_suggester",
            request_serializer=indexer__pb2.SuggesterRequest.SerializeToString,
            response_deserializer=indexer__pb2.SuggesterReply.FromString,
        )
        self.get = channel.unary_unary(
            "/iart.indexer.Indexer/get",
            request_serializer=indexer__pb2.GetRequest.SerializeToString,
            response_deserializer=indexer__pb2.GetReply.FromString,
        )
        self.search = channel.unary_unary(
            "/iart.indexer.Indexer/search",
            request_serializer=indexer__pb2.SearchRequest.SerializeToString,
            response_deserializer=indexer__pb2.SearchReply.FromString,
        )
        self.list_search_result = channel.unary_unary(
            "/iart.indexer.Indexer/list_search_result",
            request_serializer=indexer__pb2.ListSearchResultRequest.SerializeToString,
            response_deserializer=indexer__pb2.ListSearchResultReply.FromString,
        )
        self.aggregate = channel.unary_unary(
            "/iart.indexer.Indexer/aggregate",
            request_serializer=indexer__pb2.AggregateRequest.SerializeToString,
            response_deserializer=indexer__pb2.AggregateReply.FromString,
        )
        self.suggest = channel.unary_unary(
            "/iart.indexer.Indexer/suggest",
            request_serializer=indexer__pb2.SuggestRequest.SerializeToString,
            response_deserializer=indexer__pb2.SuggestReply.FromString,
        )
        self.build_indexer = channel.unary_unary(
            "/iart.indexer.Indexer/build_indexer",
            request_serializer=indexer__pb2.BuildIndexerRequest.SerializeToString,
            response_deserializer=indexer__pb2.BuildIndexerReply.FromString,
        )
        self.build_feature_cache = channel.unary_unary(
            "/iart.indexer.Indexer/build_feature_cache",
            request_serializer=indexer__pb2.BuildFeatureCacheRequest.SerializeToString,
            response_deserializer=indexer__pb2.BuildFeatureCacheReply.FromString,
        )
        self.indexing = channel.stream_stream(
            "/iart.indexer.Indexer/indexing",
            request_serializer=indexer__pb2.IndexingRequest.SerializeToString,
            response_deserializer=indexer__pb2.IndexingReply.FromString,
        )
        self.dump = channel.unary_stream(
            "/iart.indexer.Indexer/dump",
            request_serializer=indexer__pb2.DumpRequest.SerializeToString,
            response_deserializer=indexer__pb2.DumpReply.FromString,
        )
        self.load = channel.stream_stream(
            "/iart.indexer.Indexer/load",
            request_serializer=indexer__pb2.LoadRequest.SerializeToString,
            response_deserializer=indexer__pb2.LoadReply.FromString,
        )


class IndexerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def list_plugins(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def build_suggester(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def search(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def list_search_result(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def aggregate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def suggest(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def build_indexer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def build_feature_cache(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def indexing(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def dump(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def load(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_IndexerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "list_plugins": grpc.unary_unary_rpc_method_handler(
            servicer.list_plugins,
            request_deserializer=indexer__pb2.ListPluginsRequest.FromString,
            response_serializer=indexer__pb2.ListPluginsReply.SerializeToString,
        ),
        "status": grpc.unary_unary_rpc_method_handler(
            servicer.status,
            request_deserializer=indexer__pb2.StatusRequest.FromString,
            response_serializer=indexer__pb2.StatusReply.SerializeToString,
        ),
        "build_suggester": grpc.unary_unary_rpc_method_handler(
            servicer.build_suggester,
            request_deserializer=indexer__pb2.SuggesterRequest.FromString,
            response_serializer=indexer__pb2.SuggesterReply.SerializeToString,
        ),
        "get": grpc.unary_unary_rpc_method_handler(
            servicer.get,
            request_deserializer=indexer__pb2.GetRequest.FromString,
            response_serializer=indexer__pb2.GetReply.SerializeToString,
        ),
        "search": grpc.unary_unary_rpc_method_handler(
            servicer.search,
            request_deserializer=indexer__pb2.SearchRequest.FromString,
            response_serializer=indexer__pb2.SearchReply.SerializeToString,
        ),
        "list_search_result": grpc.unary_unary_rpc_method_handler(
            servicer.list_search_result,
            request_deserializer=indexer__pb2.ListSearchResultRequest.FromString,
            response_serializer=indexer__pb2.ListSearchResultReply.SerializeToString,
        ),
        "aggregate": grpc.unary_unary_rpc_method_handler(
            servicer.aggregate,
            request_deserializer=indexer__pb2.AggregateRequest.FromString,
            response_serializer=indexer__pb2.AggregateReply.SerializeToString,
        ),
        "suggest": grpc.unary_unary_rpc_method_handler(
            servicer.suggest,
            request_deserializer=indexer__pb2.SuggestRequest.FromString,
            response_serializer=indexer__pb2.SuggestReply.SerializeToString,
        ),
        "build_indexer": grpc.unary_unary_rpc_method_handler(
            servicer.build_indexer,
            request_deserializer=indexer__pb2.BuildIndexerRequest.FromString,
            response_serializer=indexer__pb2.BuildIndexerReply.SerializeToString,
        ),
        "build_feature_cache": grpc.unary_unary_rpc_method_handler(
            servicer.build_feature_cache,
            request_deserializer=indexer__pb2.BuildFeatureCacheRequest.FromString,
            response_serializer=indexer__pb2.BuildFeatureCacheReply.SerializeToString,
        ),
        "indexing": grpc.stream_stream_rpc_method_handler(
            servicer.indexing,
            request_deserializer=indexer__pb2.IndexingRequest.FromString,
            response_serializer=indexer__pb2.IndexingReply.SerializeToString,
        ),
        "dump": grpc.unary_stream_rpc_method_handler(
            servicer.dump,
            request_deserializer=indexer__pb2.DumpRequest.FromString,
            response_serializer=indexer__pb2.DumpReply.SerializeToString,
        ),
        "load": grpc.stream_stream_rpc_method_handler(
            servicer.load,
            request_deserializer=indexer__pb2.LoadRequest.FromString,
            response_serializer=indexer__pb2.LoadReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("iart.indexer.Indexer", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Indexer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def list_plugins(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/list_plugins",
            indexer__pb2.ListPluginsRequest.SerializeToString,
            indexer__pb2.ListPluginsReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def status(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/status",
            indexer__pb2.StatusRequest.SerializeToString,
            indexer__pb2.StatusReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def build_suggester(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/build_suggester",
            indexer__pb2.SuggesterRequest.SerializeToString,
            indexer__pb2.SuggesterReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/get",
            indexer__pb2.GetRequest.SerializeToString,
            indexer__pb2.GetReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def search(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/search",
            indexer__pb2.SearchRequest.SerializeToString,
            indexer__pb2.SearchReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def list_search_result(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/list_search_result",
            indexer__pb2.ListSearchResultRequest.SerializeToString,
            indexer__pb2.ListSearchResultReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def aggregate(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/aggregate",
            indexer__pb2.AggregateRequest.SerializeToString,
            indexer__pb2.AggregateReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def suggest(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/suggest",
            indexer__pb2.SuggestRequest.SerializeToString,
            indexer__pb2.SuggestReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def build_indexer(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/build_indexer",
            indexer__pb2.BuildIndexerRequest.SerializeToString,
            indexer__pb2.BuildIndexerReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def build_feature_cache(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/iart.indexer.Indexer/build_feature_cache",
            indexer__pb2.BuildFeatureCacheRequest.SerializeToString,
            indexer__pb2.BuildFeatureCacheReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def indexing(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/iart.indexer.Indexer/indexing",
            indexer__pb2.IndexingRequest.SerializeToString,
            indexer__pb2.IndexingReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def dump(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/iart.indexer.Indexer/dump",
            indexer__pb2.DumpRequest.SerializeToString,
            indexer__pb2.DumpReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def load(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/iart.indexer.Indexer/load",
            indexer__pb2.LoadRequest.SerializeToString,
            indexer__pb2.LoadReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
