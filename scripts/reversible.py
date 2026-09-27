"""
scripts/reversible.py

Memory-saving backward pass for the midpoint (leapfrog) stack.

THE IDEA
--------
The standard stack keeps every p[l] alive so autograd can compute each layer's
gradients. The midpoint rule is exactly invertible, so we do not have to:

    forward   p[l+1] = p[l-1] + 2h*f_l(p[l])
    inverse   p[l-1] = p[l+1] - 2h*f_l(p[l])

f is evaluated at p[l], which the backward walk already holds. So the forward
saves ONLY the final adjacent pair (p[L], p[L-1]) and the backward walks down,
reconstructing each p[l-1], taking that layer's gradients, and dropping the
state it no longer needs. Activation memory goes from O(n_layers) to O(1).

THE GRADIENT RECURRENCE
-----------------------
Layer l maps (p[l-1], p[l]) -> p[l+1], so it feeds two upstream gradients:

    dL/dp[l-1] += g[l+1]                       (identity path, exact)
    dL/dp[l]   += 2h * vjp(f_l, p[l], g[l+1])  (through f)

Read the other way: g[k] only ever receives from layer k (the f path) and
layer k+1 (the identity path). Walking down from l = L-1, layer k+1 is always
processed before layer k, so g[k] is COMPLETE the moment layer k is done. That
is what lets a single downward sweep be exact rather than approximate.

Seeding: the readout is final = blend*p[L] + (1-blend)*p[L-1], so
    g[L]   = blend * dL/dfinal
    g[L-1] = (1-blend) * dL/dfinal

Bootstrap: layer 0 is the ordinary residual p[1] = p[0] + f_0(p[0]) (it exists
only to manufacture the second state the two-term recurrence needs). Its rule
is never inverted -- p[0] falls out of layer 1's inverse -- but it does carry
gradient, through both its identity and its f path:

    dL/dp[0] += g[1] + vjp(f_0, p[0], g[1])

COST
----
One extra forward per layer: the backward recomputes f_l once, and that single
recomputation serves double duty -- it reconstructs p[l-1] AND provides the
graph for the vjp. Roughly 2x forward FLOPs instead of 1x, in exchange for
dropping per-layer activation storage.

EULER IS DELIBERATELY ABSENT. Step 2 showed its inverse is implicit and its
reconstruction error grows ~1.4x per layer. A reversible backward built on it
would produce silently wrong gradients, which is worse than no backward at all.
"""

import torch


class _MidpointReversible(torch.autograd.Function):
    """Midpoint stack whose backward reconstructs activations instead of storing them.

    Non-tensor args (blocks, h, blend, counts) ride along on ctx; backward
    returns None in their slots. Parameters are passed flat, in block order,
    so the returned gradient tuple lines up positionally.
    """

    @staticmethod
    def forward(ctx, p0, cos, sin, blocks, h, blend, counts, *params):
        with torch.no_grad():
            # bootstrap: ordinary residual, produces the second state
            p_prev = p0
            p_cur = p0 + blocks[0](p0, cos, sin)
            # leapfrog
            for blk in blocks[1:]:
                p_next = p_prev + 2.0 * h * blk(p_cur, cos, sin)
                p_prev, p_cur = p_cur, p_next
            final = blend * p_cur + (1.0 - blend) * p_prev

        # THE WHOLE POINT: only the final pair is kept. No per-layer states.
        ctx.save_for_backward(p_cur, p_prev, cos, sin)
        ctx.blocks = blocks
        ctx.h = h
        ctx.blend = blend
        ctx.counts = counts
        return final

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_final):
        p_next, p_cur, cos, sin = ctx.saved_tensors   # p[L], p[L-1]
        blocks, h, blend = ctx.blocks, ctx.h, ctx.blend
        L = len(blocks)

        # seed from the readout blend
        g_next = blend * grad_final                    # g[L]
        g_cur = (1.0 - blend) * grad_final             # g[L-1]

        param_grads = [None] * L

        for l in range(L - 1, 0, -1):
            blk = blocks[l]
            plist = list(blk.parameters())

            # one recomputation of f_l(p[l]) serves both jobs below
            with torch.enable_grad():
                x = p_cur.detach().requires_grad_(True)
                out = blk(x, cos, sin)

            # job 1: reconstruct p[l-1], then drop p[l+1]
            with torch.no_grad():
                p_prev = p_next - 2.0 * h * out.detach()

            # job 2: this layer's gradients
            grads = torch.autograd.grad(out, [x] + plist, grad_outputs=2.0 * h * g_next)

            g_cur_done = g_cur + grads[0]   # g[l] now complete
            param_grads[l] = list(grads[1:])
            g_prev = g_next                 # identity path: dL/dp[l-1] += g[l+1]

            p_next, p_cur = p_cur, p_prev
            g_next, g_cur = g_cur_done, g_prev
            del out, x, grads, p_prev, g_cur_done

        # bootstrap layer: p[1] = p[0] + f_0(p[0]); p_cur is now p[0]
        blk0 = blocks[0]
        plist0 = list(blk0.parameters())
        with torch.enable_grad():
            x0 = p_cur.detach().requires_grad_(True)
            out0 = blk0(x0, cos, sin)
        grads0 = torch.autograd.grad(out0, [x0] + plist0, grad_outputs=g_next)

        grad_p0 = g_cur + g_next + grads0[0]
        param_grads[0] = list(grads0[1:])

        flat = []
        for l in range(L):
            flat.extend(param_grads[l])
        # slots: p0, cos, sin, blocks, h, blend, counts, *params
        return (grad_p0, None, None, None, None, None, None, *flat)


def midpoint_reversible(p0, cos, sin, blocks, h, blend):
    """Run the midpoint stack with the reconstructing backward."""
    params, counts = [], []
    for blk in blocks:
        ps = list(blk.parameters())
        counts.append(len(ps))
        params.extend(ps)
    return _MidpointReversible.apply(p0, cos, sin, blocks, h, blend, tuple(counts), *params)
