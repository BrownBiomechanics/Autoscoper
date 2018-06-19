function [socket_descriptor] = openConnection(Address)
%OPENCONNECTION Summary of this function goes here
%   Detailed explanation goes here
socket_descriptor = tcpip(Address, 30000, 'NetworkRole', 'client');
fopen(socket_descriptor);

end

