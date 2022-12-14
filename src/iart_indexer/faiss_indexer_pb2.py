# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: faiss_indexer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13\x66\x61iss_indexer.proto\x12\x12iart.faiss_indexer\"4\n\x05Query\x12\r\n\x05value\x18\x01 \x03(\x02\x12\x0e\n\x06plugin\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\"\'\n\nCollection\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\"F\n\x0bListRequest\x12\x13\n\x0b\x63ollections\x18\x01 \x03(\t\x12\"\n\x1ainclude_default_collection\x18\x02 \x01(\x08\"@\n\tListReply\x12\x33\n\x0b\x63ollections\x18\x01 \x03(\x0b\x32\x1e.iart.faiss_indexer.Collection\"t\n\rSearchRequest\x12\x13\n\x0b\x63ollections\x18\x01 \x03(\t\x12\"\n\x1ainclude_default_collection\x18\x03 \x01(\x08\x12*\n\x07queries\x18\x02 \x03(\x0b\x32\x19.iart.faiss_indexer.Query\"\x1a\n\x0bSearchReply\x12\x0b\n\x03ids\x18\x01 \x03(\t\"&\n\x0fIndexingRequest\x12\x13\n\x0b\x63ollections\x18\x01 \x03(\t\"\x0f\n\rIndexingReply\"#\n\x0cTrainRequest\x12\x13\n\x0b\x63ollections\x18\x01 \x03(\t\"\x0c\n\nTrainReply\"1\n\rDeleteRequest\x12\x13\n\x0b\x63ollections\x18\x01 \x03(\t\x12\x0b\n\x03ids\x18\x02 \x03(\t\"\r\n\x0b\x44\x65leteReply2\x9b\x03\n\x0c\x46\x61issIndexer\x12H\n\x04list\x12\x1f.iart.faiss_indexer.ListRequest\x1a\x1d.iart.faiss_indexer.ListReply\"\x00\x12N\n\x06search\x12!.iart.faiss_indexer.SearchRequest\x1a\x1f.iart.faiss_indexer.SearchReply\"\x00\x12T\n\x08indexing\x12#.iart.faiss_indexer.IndexingRequest\x1a!.iart.faiss_indexer.IndexingReply\"\x00\x12K\n\x05train\x12 .iart.faiss_indexer.TrainRequest\x1a\x1e.iart.faiss_indexer.TrainReply\"\x00\x12N\n\x06\x64\x65lete\x12!.iart.faiss_indexer.DeleteRequest\x1a\x1f.iart.faiss_indexer.DeleteReply\"\x00\x42\x02P\x01\x62\x06proto3')



_QUERY = DESCRIPTOR.message_types_by_name['Query']
_COLLECTION = DESCRIPTOR.message_types_by_name['Collection']
_LISTREQUEST = DESCRIPTOR.message_types_by_name['ListRequest']
_LISTREPLY = DESCRIPTOR.message_types_by_name['ListReply']
_SEARCHREQUEST = DESCRIPTOR.message_types_by_name['SearchRequest']
_SEARCHREPLY = DESCRIPTOR.message_types_by_name['SearchReply']
_INDEXINGREQUEST = DESCRIPTOR.message_types_by_name['IndexingRequest']
_INDEXINGREPLY = DESCRIPTOR.message_types_by_name['IndexingReply']
_TRAINREQUEST = DESCRIPTOR.message_types_by_name['TrainRequest']
_TRAINREPLY = DESCRIPTOR.message_types_by_name['TrainReply']
_DELETEREQUEST = DESCRIPTOR.message_types_by_name['DeleteRequest']
_DELETEREPLY = DESCRIPTOR.message_types_by_name['DeleteReply']
Query = _reflection.GeneratedProtocolMessageType('Query', (_message.Message,), {
  'DESCRIPTOR' : _QUERY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.Query)
  })
_sym_db.RegisterMessage(Query)

Collection = _reflection.GeneratedProtocolMessageType('Collection', (_message.Message,), {
  'DESCRIPTOR' : _COLLECTION,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.Collection)
  })
_sym_db.RegisterMessage(Collection)

ListRequest = _reflection.GeneratedProtocolMessageType('ListRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTREQUEST,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.ListRequest)
  })
_sym_db.RegisterMessage(ListRequest)

ListReply = _reflection.GeneratedProtocolMessageType('ListReply', (_message.Message,), {
  'DESCRIPTOR' : _LISTREPLY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.ListReply)
  })
_sym_db.RegisterMessage(ListReply)

SearchRequest = _reflection.GeneratedProtocolMessageType('SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREQUEST,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.SearchRequest)
  })
_sym_db.RegisterMessage(SearchRequest)

SearchReply = _reflection.GeneratedProtocolMessageType('SearchReply', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREPLY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.SearchReply)
  })
_sym_db.RegisterMessage(SearchReply)

IndexingRequest = _reflection.GeneratedProtocolMessageType('IndexingRequest', (_message.Message,), {
  'DESCRIPTOR' : _INDEXINGREQUEST,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.IndexingRequest)
  })
_sym_db.RegisterMessage(IndexingRequest)

IndexingReply = _reflection.GeneratedProtocolMessageType('IndexingReply', (_message.Message,), {
  'DESCRIPTOR' : _INDEXINGREPLY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.IndexingReply)
  })
_sym_db.RegisterMessage(IndexingReply)

TrainRequest = _reflection.GeneratedProtocolMessageType('TrainRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRAINREQUEST,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.TrainRequest)
  })
_sym_db.RegisterMessage(TrainRequest)

TrainReply = _reflection.GeneratedProtocolMessageType('TrainReply', (_message.Message,), {
  'DESCRIPTOR' : _TRAINREPLY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.TrainReply)
  })
_sym_db.RegisterMessage(TrainReply)

DeleteRequest = _reflection.GeneratedProtocolMessageType('DeleteRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETEREQUEST,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.DeleteRequest)
  })
_sym_db.RegisterMessage(DeleteRequest)

DeleteReply = _reflection.GeneratedProtocolMessageType('DeleteReply', (_message.Message,), {
  'DESCRIPTOR' : _DELETEREPLY,
  '__module__' : 'faiss_indexer_pb2'
  # @@protoc_insertion_point(class_scope:iart.faiss_indexer.DeleteReply)
  })
_sym_db.RegisterMessage(DeleteReply)

_FAISSINDEXER = DESCRIPTOR.services_by_name['FaissIndexer']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'P\001'
  _QUERY._serialized_start=43
  _QUERY._serialized_end=95
  _COLLECTION._serialized_start=97
  _COLLECTION._serialized_end=136
  _LISTREQUEST._serialized_start=138
  _LISTREQUEST._serialized_end=208
  _LISTREPLY._serialized_start=210
  _LISTREPLY._serialized_end=274
  _SEARCHREQUEST._serialized_start=276
  _SEARCHREQUEST._serialized_end=392
  _SEARCHREPLY._serialized_start=394
  _SEARCHREPLY._serialized_end=420
  _INDEXINGREQUEST._serialized_start=422
  _INDEXINGREQUEST._serialized_end=460
  _INDEXINGREPLY._serialized_start=462
  _INDEXINGREPLY._serialized_end=477
  _TRAINREQUEST._serialized_start=479
  _TRAINREQUEST._serialized_end=514
  _TRAINREPLY._serialized_start=516
  _TRAINREPLY._serialized_end=528
  _DELETEREQUEST._serialized_start=530
  _DELETEREQUEST._serialized_end=579
  _DELETEREPLY._serialized_start=581
  _DELETEREPLY._serialized_end=594
  _FAISSINDEXER._serialized_start=597
  _FAISSINDEXER._serialized_end=1008
# @@protoc_insertion_point(module_scope)
