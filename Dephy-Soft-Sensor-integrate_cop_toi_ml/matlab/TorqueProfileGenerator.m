t0 = 0;
t1 = 20;
t2 = 38;
t3 = 48;
ts = 0.056;
npt = 0.025/3;
user_mass = 50;

tp = user_mass * npt;

a1 = (2 * (tp - ts)) / (t1 - t2)^3;

b1 = -(3 * (t1 + t2) * (tp - ts)) / ((t1 - t2)^3);
c1 = (6 * t1 * t2 * (tp - ts)) / ((t1 - t2)^3);
d1 = -((-(t1^3) * tp + 3 * (t1^2) * t2 * tp - 3 * t1 * (...
    t2^2) * ts + (t2^3) * ts) / ((t1 - t2)^3));
a2 = -((tp - ts) / (2 * ((t2 - t3)^3)));
b2 = (3 * t3 * (tp - ts)) / (2 * ((t2 - t3)^3));
c2 = (3 * ((t2^2) - 2 * t2 * t3) * (tp - ts)) / (...
            2 * ((t2 - t3)^3));
d2 = -((3 * (t2^2) * t3 * tp - 6 * t2 * (t3^2) * tp + 2 * (...
    t3^3) * tp - 2 * (t2^3) * ts + 3 * (t2^2) * t3 * ts) / (...
                        2 * ((t2 - t3)^3)));


%%

percent_gait = 0:.1:100;
tau = zeros(1,1001);
for i = 1:1:1001
    if and((percent_gait(i) <= t1), (t0 <= percent_gait(i)))
        tau(i) = ts / (t1 - t0) * percent_gait(i) - ts / (t1 - t0) * t0;
    elseif percent_gait(i) <= t2
        tau(i) = a1 * (percent_gait(i)^3) + b1 * (percent_gait(i)^2) + c1 * percent_gait(i) + d1;
    elseif percent_gait(i) <=t3
        tau(i) = a2 * (percent_gait(i)^3) + b2 * (percent_gait(i)^2) + c2 * percent_gait(i) + d2;
    else
        tau(i)=0;
    end
end


plot(percent_gait,tau)
xlabel('Percent Gait Cycle')
ylabel('Torque (Nm/kg)')