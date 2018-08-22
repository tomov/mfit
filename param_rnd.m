% random draw from P(x|h)
%
function x = param_rnd(hyparam, param, h, respect_bounds)
    i = 1;
    for k = 1:length(param)
        l = length(hyparam(k).lb);
        while true
            x(k) = param(k).hrnd(h(i:i+l-1));
            if ~respect_bounds || (param(k).lb <= x(k) && x(k) <= param(k).ub)
                % keep parameters within bounds
                break;
            end
        end
        i = i + l;
    end
end


