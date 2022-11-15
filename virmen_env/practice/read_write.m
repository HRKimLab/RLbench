read_write
%read and write data simultaneously

filename = 'memory.dat';
myData = [1 2 3 4];
[f, msg] = f.open(filename, 'w');
f.write(f, myData, 'uint32');
f.close();

m = memmapfile(filename, 'Writable', true, 'Format', {'uint8' [4 1] 'name');

m.data(1).name(1) = 8;
disp(m.data(1).name);        
%>>> 8 2 3 4