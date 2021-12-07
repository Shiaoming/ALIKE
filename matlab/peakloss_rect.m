clear;
close all;

x = -1:0.01:1;

p0 = 0.5;
p1 = -0.5;

d = abs(x - p0);

c0 = 2 .* (x>=-0.75 & x <= -0.25);
c1 = 2 .* (x>=0.25 & x <= 0.75);
c2 = 1.25 .* (x>=0.1 & x <= 0.9);

peak_loss0 = sum(d.*c0) / length(x)
peak_loss1 = sum(d.*c1) / length(x)
peak_loss2 = sum(d.*c2) / length(x)

createfigure(x, [c0;c1;c2], d, peak_loss0,peak_loss1, peak_loss2);