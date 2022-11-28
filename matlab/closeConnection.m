function closeConnection(autoscoper_socket)
%CLOSECONNECTION Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[13]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end
