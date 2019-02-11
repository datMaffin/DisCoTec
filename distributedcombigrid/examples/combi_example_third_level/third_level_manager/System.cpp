#include "System.hpp"

System::System(const std::string& name,
               const AmqpClient::Channel::ptr_t& channel,
               const ServerSocket& server) : name_(name)
{
  createMessageQueues(channel);
  createDataConnection(server, channel);
}

/*
 * Creates queues in both directions to communicate with the system.
 * Queues are non passive, non durable, non exclusive and delete themselves
 * automatically when the last using application dies.
 */
void System::createMessageQueues(AmqpClient::Channel::ptr_t channel)
{
  inQueue_ = MessageUtils::createMessageQueue(name_+"_in", channel);
  outQueue_ = MessageUtils::createMessageQueue(name_+"_in", channel);
}

// TODO initialization which ensures that remote server accepts.
void System::createDataConnection(const ServerSocket& server,
                                  const AmqpClient::Channel::ptr_t& channel)
{
  std::string message;
  receiveMessage(channel, message); // waits until system is ready
  assert(message == "create_data_conn");
  dataConnection_ = std::shared_ptr<ClientSocket>(server.acceptClient());
}

void System::sendMessage(const std::string& message, AmqpClient::Channel::ptr_t channel)
{
  MessageUtils::sendMessage(message, inQueue_, channel);
}

bool System::receiveMessage(AmqpClient::Channel::ptr_t channel, std::string& message, int timeout)
{
  return MessageUtils::receiveMessage(channel, outQueue_, message);
}

std::string System::getName() const
{
  return name_;
}

std::shared_ptr<ClientSocket> System::getDataConnection() const
{
  return dataConnection_;
}
