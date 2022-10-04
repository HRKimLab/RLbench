function send_answer
%send and answer simultaneously

filename = fullfile(tempdir, 'talk_answer.dat');

m = memmapfile(filename, 'Writable', true, 'Format', 'uint8');

while true
    flag = input('input or output (input: i, output: o, exit: e): ','s');
    
    if (flag == 'i')
        