% random draw from P(h)
%
function h = hyparam_rnd(hyparam, param)
    h = [];
    for k = 1:length(hyparam)
        while true
            h_k = hyparam(k).rnd();
            x_k = param(k).hrnd(h_k);
            logp = param(k).hlogpdf(x_k, h_k);
            if all(hyparam(k).lb <= h_k) && all(h_k <= hyparam(k).ub) && ~isinf(logp) && ~isnan(logp)
                % keep hyperparameters within bounds and do a sanity check
                % to make sure they don't generate invalid parameters
                h = [h h_k];
                break;
            end
        end
    end
end


