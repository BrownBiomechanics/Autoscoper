function [ncc_out] = getNCC_Sum(pose,autoscoper_socket,volume)

% volume = 1;
%open Connection
% autoscoper_socket = openConnection('127.0.0.1');


%GETNCC Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[8 typecast(int32(volume),'uint8') typecast(double(pose),'uint8')]);
while autoscoper_socket.BytesAvailable == 0
    pause(0.01)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);

%-99999 is nsot tracked
%returned values is 1 - ncc, so 0 is best, 2 is worst
ncc = zeros(1,data(2));
for i = 1:data(2)
    val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double') ;
    if(val < 0)
        val = 1e3;
    end
    ncc(i) = val;
end

ncc_out  = ncc(1) + ncc(2);

% fclose(autoscoper_socket);

