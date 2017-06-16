#include "mpi_helper.h"
#include "constants.h"

void MPISendLite(Message &msg, int target, MessageType type)
{
	string out;
	msg.SerializeToString(&out);
	MPI_Send(out.data(), out.size(), MPI_CHAR, target, static_cast<int>(type), MPI_COMM_WORLD);
}

MPIMsgDescriptor MPIRecvLite(char *buf, int type, int source)
{
	MPI_Status status;
	MPI_Recv(buf, MAX_LENGTH, MPI_CHAR, source, type, MPI_COMM_WORLD, &status);
	int length;
	MPI_Get_count(&status, MPI_CHAR, &length);

	MPIMsgDescriptor ret(length, status.MPI_SOURCE, static_cast<MessageType>(status.MPI_TAG));
	return ret;
}