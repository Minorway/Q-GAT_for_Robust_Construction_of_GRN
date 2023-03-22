import torch
from torch import Tensor
# from torch_scatter import scatter
from typing import Optional
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import Linear
from torch_scatter import scatter
# from GQT_module.QuadraticOperation import QuadraticOperation
from .QuadraticOperation import QuadraticOperation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Graph_Quadratic_Pooling(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    x1=scatter(x, batch, dim=-2, dim_size=size, reduce='mean')
    Q_layer1=QuadraticOperation(x1.size(-1),x1.size(-1)).to(device)
    x2=Q_layer1(x1)
    x_s=(x1+x2)/2
    return x_s



