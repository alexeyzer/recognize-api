# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from gen.python.api.recognize.v1 import recognize_api_pb2 as api_dot_recognize_dot_v1_dot_recognize__api__pb2


class RecognizeApiServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.recognizePhoto = channel.unary_unary(
                '/recognize.api.RecognizeApiService/recognizePhoto',
                request_serializer=api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoRequest.SerializeToString,
                response_deserializer=api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoResponse.FromString,
                )


class RecognizeApiServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def recognizePhoto(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RecognizeApiServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'recognizePhoto': grpc.unary_unary_rpc_method_handler(
                    servicer.recognizePhoto,
                    request_deserializer=api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoRequest.FromString,
                    response_serializer=api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'recognize.api.RecognizeApiService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RecognizeApiService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def recognizePhoto(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/recognize.api.RecognizeApiService/recognizePhoto',
            api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoRequest.SerializeToString,
            api_dot_recognize_dot_v1_dot_recognize__api__pb2.recognizePhotoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
