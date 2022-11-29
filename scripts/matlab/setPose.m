function setPose(autoscoper_socket,volume,frame, pose)
%SETPOSE Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[7 typecast(int32(volume),'uint8') typecast(int32(frame),'uint8') typecast(double(pose),'uint8')]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

