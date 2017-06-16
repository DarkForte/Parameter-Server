#pragma once
#include <mpi.h>
#include <google/protobuf/message.h>
#include <iostream>
#include "message_type.h"

using google::protobuf::Message;
using std::string;

class MPIMsgDescriptor
{
public:
	int length;
	int source;
	MessageType type;

	MPIMsgDescriptor(int length, int source, MessageType type)
	{
		this->length = length;
		this->source = source;
		this->type = type;
	}
};

void MPISendLite(Message &msg, int target, MessageType type);
MPIMsgDescriptor MPIRecvLite(char *buf, int type, int source);