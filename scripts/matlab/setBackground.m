function setBackground(autoscoper_socket,value)
%SETBACKGROUND Summary of this function goes here
%   Detailed explanation goes here
%set background
fwrite(autoscoper_socket,[9 typecast(double(value),'uint8')]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

