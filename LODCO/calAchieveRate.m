function r = calAchieveRate(h, p, omega, sigma)
%% calculate achievable rate
r = omega*log2(1+h*p/sigma);

end