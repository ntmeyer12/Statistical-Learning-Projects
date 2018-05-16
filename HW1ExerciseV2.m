clear all, clc

load('TrainingSamplesDCT_8.mat')
[m,n] = size(TrainsampleDCT_BG);
[a,b] = size(TrainsampleDCT_FG);

zig=load('Zig-Zag Pattern.txt');
zig=zig+1;

% This is for the Background, which has a 1053 rows, estimated
% Py(grass)=0.8081
for j=1:m
    M = abs(TrainsampleDCT_BG(j,2:end));
    [val, idx] = max(M);
    BG(j) = idx+1;
end

[counts_bg, bins] = hist(BG, 1:1:40);
figure(1), bar(bins, counts_bg);
xlabel('X (DCT Index)'), ylabel('Count'), title('P(x|grass) Histogram')
for i = 1:numel(bins)
    text(bins(i) - 0.2, counts_bg(i) + 30, [sprintf('%.1f',...
        (counts_bg(i)/1053)*100) '%'], 'VerticalAlignment', 'top', 'FontSize', 8)
end

for i=1:40
    BDR_grass(i,1) = double(counts_bg(i)/1053*0.8081);
end
%----------------------------------------------------------------%


% This is for the Foreground, which has 250 rows, estimated
% Py(cheetah)=0.1919
for i=1:a
   M = abs(TrainsampleDCT_FG(i,2:end));
   [val, idx] = max(M);
   FG(i) = idx+1;
end

[counts_fg, bins] = hist(FG,1:1:40);
figure(2), bar(bins, counts_fg)
xlabel('X (DCT Index)'), ylabel('Count'), title('P(x|cheetah) Histogram')
for i = 1:numel(bins)
    text(bins(i) - 0.2, counts_fg(i) + 8, [sprintf('%.1f',...
        (counts_fg(i)/250)*100) '%'], 'VerticalAlignment', 'top', 'FontSize', 8)
end

for i=1:40
    BDR_cheetah(i,1) = double(counts_fg(i)/250*0.1919);
end
% histf(FG,x,'facealpha', 0.5)
% legalpha('BG','FG','location','northwest')

%----------------------------------------------------------------%


% BDR if P(x|grass)P(grass) < P(x|cheetah)P(cheetah), choose cheetah, ie,
% add to the index array.
k=1;

for i=1:40
    if BDR_grass(i)<BDR_cheetah(i)
        bin(k)=i;
        k=k+1;
    end
end

% read the image of the cheetah
Ch = imread('cheetah.bmp');
Ch = im2double(Ch);
[x,y] = size(Ch);


% for each block, perform dct2 and do zig-zag operation
count = 1;
for i=1:x-7
    for j=1:y-7
        blk = Ch(i:i+7,j:j+7);
        blkdct2 = dct2(blk);
        blkdct2_reordered(zig)=blkdct2;
        blkzz(count,:) = blkdct2_reordered;
        count = count + 1;
    end
end

[row, col] = size(blkzz);

% for each row, get second highest magnitude
for j=1:row
    M = abs(blkzz(j,2:end));
    [val, idx] = max(M);
    cheetah(j) = idx+1;
end
idx = 1;

Seg(255,270)=0;
% apply to separated foreground and background using results from BDR
% AKA this is our classifier
for i=1:x-7
    for j=1:y-7
        if ismember(cheetah(idx),bin)
            Seg(i,j) = 1;
        else
            Seg(i,j) = 0;
        end
        idx = idx + 1;
    end
end

figure;
imagesc(Seg)
colormap(gray(255))

Ch_mask=imread('cheetah_mask.bmp');
Ch_adj = double(Ch_mask)/255;

errors=0;
for x = 1:255
    for y=1:270
        if Ch_adj(x,y)~=Seg(x,y)
            errors=errors+1;
        end
    end
end

p_err = errors/(255*270)*100;
error = ['8-Dimensional Error: ' num2str(p_err) '%'];
disp(error);

det = 0;
fal=0;
for x = 1:255
    for y = 1:270
        if Ch_adj(x,y) == 1 && Seg(x,y) == 1
            det = det+1;
        end
        if Ch_adj(x,y) == 0 && Seg(x,y) == 1
            fal = fal+1;
        end
    end
end

detection_rate = det/(255*270*0.1919);
false_alarm = fal / (255*270*0.8081);
f_rate = ['8-Dimensional False Alarm Rate: ' num2str(false_alarm)];
drate = ['8-Dimensional Detection Rate: ' num2str(detection_rate)];
disp(drate);
disp(f_rate);

