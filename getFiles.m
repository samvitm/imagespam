function ret = getFiles()   
    clc
    disp('Getting Spam Images : ');
    spams=dir('./spam/');   
    spaminfo = [];
    for i = 1:length(spams)
        if ~spams(i).isdir
            filepath = strcat('./spam/',char(spams(i).name));    
            info = getinfo(filepath);                
            disp(info);    
            spaminfo = [spaminfo; info];
        end
    end
    disp('Getting Natural Images : ');
    natimgs=dir('./naturals/');   
    natinfo = [];
    for i = 1:length(natimgs)
        if ~natimgs(i).isdir
            filepath = strcat('./naturals/',char(natimgs(i).name));
            info = getinfo(filepath);                
            disp(info);    
            natinfo = [natinfo; info];
        end
    end
    [matrix,classes] = getFeatureMatrix(spaminfo,natinfo);
    disp(matrix);
    trainedData = train(matrix,classes);
    newData = getTestData();
    newClasses = classify(trainedData,newData);
    disp('New Classes : ');
    disp(newClasses);
    ret = 'Processed';    
end

function testdata = getTestData()
    timgs  = dir('./tests/');
    tinfo = [];
    for i = 1:length(timgs)
        if ~timgs(i).isdir
            filepath = strcat('./tests/',char(timgs(i).name));    
            info = getinfo(filepath);                
            %disp(info);    
            tinfo = [tinfo; info];
        end
    end
    testdata = getFeatureMatrix(tinfo,[]);    
end

function hexval = convert2hex(rgb)
    rn = dec2hex(rgb(1,1));
    gn = dec2hex(rgb(1,2));
    bn = dec2hex(rgb(1,3));
    hexval = strcat(char(rn),char(gn),char(bn));
end

function format = getFormat(form)
    if strcmp(form,'jpg')
        format = 1;    
    elseif strcmp(form,'tiff')
           format = 2;        
    elseif strcmp(form,'png')
        format = 3;        
    elseif strcmp(form,'gif')
        format = 4;        
    elseif strcmp(form,'bmp')
        format = 5;
    else
        format = -1;
    end    
end

function [fmat,class] = getFeatureMatrix(spaminfo,natinfo)
    % This function creates a M x N size matrix which wil be used as a
    % training data. Will contain all the data points reqired
    % - Each row is an observations.
    % - Each column is a feature.
    % - Format = [ 1:jpg 2:tiff 3:png ...]
    %  FileSize | Width | Height | BitDepth | Format | AverageRGB | MostFrequent Color | ColorFrequency 
    %  Class :  0 = Spam , 1 = NotSpam
    
    fmat = [];
    fmat = double(fmat);
    class = [];
    for i = 1:length(spaminfo)
        e = spaminfo(i);
        entry = double([e.FileSize e.Width e.Height e.BitDepth getFormat(e.Format) e.AverageRGB e.MostFrequentColor e.ColorFrequency]);
        fmat = [fmat ; entry];
        class = [class;0];       
    end
    
    for i = 1:length(natinfo)
        e = natinfo(i);
        entry = double([e.FileSize e.Width e.Height e.BitDepth getFormat(e.Format) e.AverageRGB e.MostFrequentColor e.ColorFrequency]);
        fmat = [fmat ; entry];
        class = [class;1];
    end    
end

function info = getinfo(img)
    file = img;
    info = imfinfo(file);
    img = imread(file);
    avrg = mean(img(:));
    info.('AverageRGB') = double(avrg);    
    rgb_columns = reshape(img, [], 3);    
    [unique_colors, m, n] = unique(rgb_columns, 'rows');
    color_counts = accumarray(n, 1);
    [max_count, idx] = max(color_counts);
    cc = [unique_colors(idx,1) unique_colors(idx,2) unique_colors(idx,3)];
    info.('MostFrequentColor') = double(cc);
    info.('ColorFrequency') = double(max_count);
end

function SVMstruct = train(fmat,class)
    SVMstruct = svmtrain(fmat,class);    
end

function newClasses = classify(SVMstruct,newData)
    newClasses = svmclassify(SVMstruct,newData);    
end