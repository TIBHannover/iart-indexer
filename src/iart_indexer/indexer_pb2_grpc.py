# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import indexer_pb2 as indexer__pb2


class IndexerStub(object):
    """python -m grpc_tools.protoc -I../web --python_out=. --grpc_python_out=.
    ../web/tunnel.proto python -m grpc_tools.protoc -I../backend --python_out=.
    --grpc_python_out=. ../backend/tunnel.proto create json stringds

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.list_plugins = channel.unary_unary(
            "/Indexer/list_plugins",
            request_serializer=indexer__pb2.ListPluginsRequest.SerializeToString,
            response_deserializer=indexer__pb2.ListPluginsReply.FromString,
        )
        self.indexing = channel.unary_unary(
            "/Indexer/indexing",
            request_serializer=indexer__pb2.IndexingRequest.SerializeToString,
            response_deserializer=indexer__pb2.IndexingReply.FromString,
        )
        self.status = channel.unary_unary(
            "/Indexer/status",
            request_serializer=indexer__pb2.StatusRequest.SerializeToString,
            response_deserializer=indexer__pb2.StatusReply.FromString,
        )
        self.build_suggester = channel.unary_unary(
            "/Indexer/build_suggester",
            request_serializer=indexer__pb2.SuggesterRequest.SerializeToString,
            response_deserializer=indexer__pb2.SuggesterReply.FromString,
        )
        self.get = channel.unary_unary(
            "/Indexer/get",
            request_serializer=indexer__pb2.GetRequest.SerializeToString,
            response_deserializer=indexer__pb2.GetReply.FromString,
        )
        self.search = channel.unary_unary(
            "/Indexer/search",
            request_serializer=indexer__pb2.SearchRequest.SerializeToString,
            response_deserializer=indexer__pb2.SearchReply.FromString,
        )
        self.list_search_result = channel.unary_unary(
            "/Indexer/list_search_result",
            request_serializer=indexer__pb2.ListSearchResultRequest.SerializeToString,
            response_deserializer=indexer__pb2.ListSearchResultReply.FromString,
        )
        self.suggest = channel.unary_unary(
            "/Indexer/suggest",
            request_serializer=indexer__pb2.SuggestRequest.SerializeToString,
            response_deserializer=indexer__pb2.SuggestReply.FromString,
        )
        self.build_indexer = channel.unary_unary(
            "/Indexer/build_indexer",
            request_serializer=indexer__pb2.IndexerRequest.SerializeToString,
            response_deserializer=indexer__pb2.IndexerReply.FromString,
        )


class IndexerServicer(object):
    """python -m grpc_tools.protoc -I../web --python_out=. --grpc_python_out=.
    ../web/tunnel.proto python -m grpc_tools.protoc -I../backend --python_out=.
    --grpc_python_out=. ../backend/tunnel.proto create json stringds

    """

    def list_plugins(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def indexing(self, request, context):
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


def add_IndexerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "list_plugins": grpc.unary_unary_rpc_method_handler(
            servicer.list_plugins,
            request_deserializer=indexer__pb2.ListPluginsRequest.FromString,
            response_serializer=indexer__pb2.ListPluginsReply.SerializeToString,
        ),
        "indexing": grpc.unary_unary_rpc_method_handler(
            servicer.indexing,
            request_deserializer=indexer__pb2.IndexingRequest.FromString,
            response_serializer=indexer__pb2.IndexingReply.SerializeToString,
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
        "suggest": grpc.unary_unary_rpc_method_handler(
            servicer.suggest,
            request_deserializer=indexer__pb2.SuggestRequest.FromString,
            response_serializer=indexer__pb2.SuggestReply.SerializeToString,
        ),
        "build_indexer": grpc.unary_unary_rpc_method_handler(
            servicer.build_indexer,
            request_deserializer=indexer__pb2.IndexerRequest.FromString,
            response_serializer=indexer__pb2.IndexerReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("Indexer", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Indexer(object):
    """python -m grpc_tools.protoc -I../web --python_out=. --grpc_python_out=.
    ../web/tunnel.proto python -m grpc_tools.protoc -I../backend --python_out=.
    --grpc_python_out=. ../backend/tunnel.proto create json stringds

    """

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
            "/Indexer/list_plugins",
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
    def indexing(
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
            "/Indexer/indexing",
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
            "/Indexer/status",
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
            "/Indexer/build_suggester",
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
            "/Indexer/get",
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
            "/Indexer/search",
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
            "/Indexer/list_search_result",
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
            "/Indexer/suggest",
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
            "/Indexer/build_indexer",
            indexer__pb2.IndexerRequest.SerializeToString,
            indexer__pb2.IndexerReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

