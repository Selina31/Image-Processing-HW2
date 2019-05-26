function accum = circ_hough(img)
%CICR_HOUGH Summary of this function goes here
%   Detailed explanation goes here
accum = zeros(540,720,3);

i_rng = size(img, 1);
j_rng = size(img, 2);


for i=1:i_rng
    for j=1:j_rng
        if img(i,j) > 0
            for r=48:2:52
                for t=0:359
                    x = i-r*cos(t*pi/180);
                    y = j-r*sin(t*pi/180);
                    if(round(x)>0 & round(x)<=size(img,1) & round(y)>0 & round(y)<=size(img,2))
                        accum(round(x), round(y), (r-46)/2) = accum(round(x), round(y), (r-46)/2)+1;
                    end
                end
            end
        end
    end
end

end

