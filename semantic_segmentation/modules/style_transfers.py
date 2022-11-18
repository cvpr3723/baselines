import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import kornia as K

def eig_val_decomp_grad_contains_nan(x: torch.Tensor) -> bool:
  """ Check if the backward pass of the eig val decomposition leads to valid gradients.

  In case two or more eigenvalues are (nearly) identical the gradient will be numerically unstable.
  This leads to nan values.

  Please find more information here (Warning):
  - https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html 
  
  Args:
      x (torch.Tensor): Compute the eig val decomposition based on this tensor of shape [B x C x C]

  Returns:
      bool: True := Gradient contains nan values, False := Gradient does NOT contain nan values.
  """
  x_ = x.detach().clone()
  x_.requires_grad = True

  _, v_ = torch.linalg.eigh(x_)
  s_ = v_.sum()
  s_.backward()

  grad_contains_nans = x_.grad.isnan().any()

  return grad_contains_nans

def transfer_global_statistics(src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
  """ Transfer the statistics from the src tensor to the target tensor.

  Accordingly the output tensor has the statistics of the src tensor.

  Args:
      src (torch.Tensor): Source tensor of shape [B x C x H1 x W1]
      trg (torch.Tensor): Target tensor of shape [B x C x H2 x W2] 

  Returns:
      torch.Tensor: Target tensor with statistics of src tensor.
  """ 
  assert (not src.isnan().any()), "Invalid input the 'src' tensor contains 'nan' values."
  assert (not trg.isnan().any()), "Invalid input the 'trg' tensor contains 'nan' values."

  batch_size, chans, _, _ = src.shape
  trg_height, trg_width = trg.shape[2:]

  src_flattened = src.view(batch_size, chans, -1) # [B x C x H1*W1]
  trg_flattened = trg.view(batch_size, chans, -1) # [B x C x H2*W2]
  
  src_mean = torch.mean(src_flattened, dim=-1, keepdim=True) # [B, C, 1]
  trg_mean = torch.mean(trg_flattened, dim=-1, keepdim=True) # [B, C, 1]

  src_reduced = src_flattened - src_mean # [B x C x H1*W1]
  trg_reduced = trg_flattened - trg_mean # [B x C x H2*W2]

  src_cov_mat = torch.bmm(src_reduced, src_reduced.transpose(1, 2)) / (src_reduced.shape[-1] - 1)# [B x C x C]
  trg_cov_mat = torch.bmm(trg_reduced, trg_reduced.transpose(1, 2)) / (trg_reduced.shape[-1] - 1) # [B x C x C]
  
  src_eigvals, src_eigvecs = torch.linalg.eigh(src_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
  src_eigvals = torch.clamp(src_eigvals, min=1e-8, max=float(torch.max(src_eigvals))) # valid op since covmat is positive (semi-)definit
  src_eigvals_sqrt = torch.sqrt(src_eigvals).unsqueeze(2) # [B, C, 1]

  trg_eigvals, trg_eigvecs = torch.linalg.eigh(trg_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
  trg_eigvals = torch.clamp(trg_eigvals, min=1e-8, max=float(torch.max(trg_eigvals))) # valid op since covmat is positive (semi-)definit
  trg_eigvals_sqrt = torch.sqrt(trg_eigvals).unsqueeze(2) # [B, C, 1]

  # transfer color statistics form source to target
  W_trg = torch.bmm(trg_eigvecs, (1 / trg_eigvals_sqrt) * trg_eigvecs.transpose(1, 2))
  trg_white = torch.bmm(W_trg, trg_reduced)

  W_src_inv = torch.bmm(src_eigvecs, src_eigvals_sqrt * src_eigvecs.transpose(1, 2))
  trg_transformed = torch.bmm(W_src_inv, trg_white) + src_mean

  trg_transformed = trg_transformed.view(batch_size, chans, trg_height, trg_width)

  alpha = torch.rand((batch_size, 1, 1, 1), device=trg_transformed.device)
  alpha = torch.clamp(alpha, min=0.0, max=0.95)

  trg_transformed = (alpha * trg) + ((1 - alpha) * trg_transformed)

  return trg_transformed

T = torch.Tensor
def transfer_local_statistics(src: T, src_anno: T, trg: T, trg_anno: T) -> torch.Tensor:
  """ Transfer local statistics from the src tensor to the target tensor.

  We transfer the statistics of the class with id = 0 (i.e. soil) and of the classes with id = {1,2} (i.e. vegetation).

  Args:
      src (torch.Tensor): Source tensor of shape [B x C x H1 x W1]
      src_anno (torch.Tensor): corresponding annotations of shape [B x h1 x w1]
      trg (torch.Tensor): change the statistics of this feature volume ... of shape [B x C x H2 x W2]
      trg_anno (torch.Tensor): corresponding annotations ... of shape [B x h2 x w2]

  Returns:
      torch.Tensor: trg with local statistics of src
  """
  blur_src = transforms.GaussianBlur(3, sigma=(1.00))
  blur_trg = transforms.GaussianBlur(5, sigma=(1.25))

  trg_new = trg.clone()
  for batch_index in range(trg.shape[0]):
    for category_id in [0, 1]:
      if category_id == 0:
        # soil mask
        src_mask = (src_anno[batch_index] == category_id)
        trg_mask = (trg_anno[batch_index] == category_id)
      else: 
        # vegetation mask
        src_mask = (src_anno[batch_index] >= category_id) & (src_anno[batch_index] != 255)
        trg_mask = (trg_anno[batch_index] >= category_id) & (trg_anno[batch_index] != 255)
      
      if (src_mask.sum() < 16):
        print(f"Skip local style transfer due to (too) low number of samples for category: {category_id} in SRC batch id {batch_index}")
        continue
      if (trg_mask.sum() < 16):
        print(f"Skip local style transfer due to (too) low number of samples for category: {category_id} in TRG batch id {batch_index}")
        continue
      
      # src[batch_index] = blur_src(src[batch_index])
      src_data = src[batch_index][:, src_mask]  # [C x M]
      trg_data = trg[batch_index][:, trg_mask]  # [C x N]

      src_mean = torch.mean(src_data, dim=-1, keepdim=True)  # [C, 1]
      trg_mean = torch.mean(trg_data, dim=-1, keepdim=True)  # [C, 1]

      src_reduced = src_data - src_mean  # [C x M]
      trg_reduced = trg_data - trg_mean  # [C x N]

      src_cov_mat = torch.mm(src_reduced, src_reduced.transpose(0, 1)) / (src_reduced.shape[-1] - 1)  # [C x C]
      trg_cov_mat = torch.mm(trg_reduced, trg_reduced.transpose(0, 1)) / (trg_reduced.shape[-1] - 1)  # [C x C]

      src_eigvals, src_eigvecs = torch.linalg.eigh(src_cov_mat)  # eigval -> [C], eigvecs -> [C, C]
      src_eigvals = torch.clamp(src_eigvals, min=1e-8, max=float(torch.max(src_eigvals)))  # valid op since covmat is positive semi-definit
      src_eigvals_sqrt = torch.sqrt(src_eigvals).unsqueeze(1)  # [C, 1]

      trg_eigvals, trg_eigvecs = torch.linalg.eigh(trg_cov_mat)  # eigval -> [C], eigvecs -> [C, C]
      trg_eigvals = torch.clamp(trg_eigvals, min=1e-8, max=float(torch.max(trg_eigvals)))  # valid op since covmat is positive semi-definit
      trg_eigvals_sqrt = torch.sqrt(trg_eigvals).unsqueeze(1)  # [C, 1]

      # transfer color statistics form source to target
      W_trg = torch.mm(trg_eigvecs, (1 / trg_eigvals_sqrt) * trg_eigvecs.transpose(0, 1))
      trg_white = torch.mm(W_trg, trg_reduced)  # [C x N]

      W_src_inv = torch.mm(src_eigvecs, src_eigvals_sqrt * src_eigvecs.transpose(0, 1))
      trg_transformed = torch.mm(W_src_inv, trg_white) + src_mean # [C x N]

      alpha = random.uniform(0.0, 0.95)
      trg_transformed =  (alpha * trg_data) + ((1 - alpha) * trg_transformed)

      trg_new[batch_index][:, trg_mask] = trg_transformed
    
    blurred = blur_trg(trg_new[batch_index])
    edges = K.filters.sobel((trg_anno[batch_index] > 0).unsqueeze(0).unsqueeze(0).float())
    edges = edges > 0.1
    edges = K.morphology.dilation(edges, torch.ones(3, 3))
    edges = edges.squeeze().bool()

    trg_new[batch_index] = (blurred * edges) + (trg_new[batch_index] * ~edges)
  
  return trg_new
