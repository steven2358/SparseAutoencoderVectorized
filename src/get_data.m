function patches = get_data()
% Returns 10000 patches for training. Checks if data is in '/data' folder 
% and downloads from remote repository if necessary.

numpatches = 10000;

url_data = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
% url_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';

data_dir = '../data/';

%% load data

if (exist([data_dir 'train-images-idx3-ubyte'],'file') ~= 2)
    t1 = tic;
    fprintf('Downloading image data... ');
    gunzip(url_data,data_dir);
    fprintf('%.2fs.\n',toc(t1));
end

% if (exist([data_dir 'train-labels-idx1-ubyte'],'file') ~= 2)
%     t2 = tic;
%     fprintf('Downloading labels... ');
%     gunzip(url_labels,data_dir)
%     fprintf('done.\n');
%     fprintf('%.2fs.\n',toc(t2));
% end

t3 = tic;
fprintf('Loading images... ');
images = loadMNISTImages('../data/train-images-idx3-ubyte');
patches = images(:,1:numpatches);
fprintf('%.2fs.\n',toc(t3));
