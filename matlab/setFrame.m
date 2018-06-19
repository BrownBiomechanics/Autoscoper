function setFrame(autoscoper_socket,frame)
%SETFRAME Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[5 typecast(int32(frame),'uint8')]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

