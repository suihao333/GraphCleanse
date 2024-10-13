def prepare_dataset_x(dataset, max_degree=0):
  # if dataset[0].x is None:
  if max_degree == 0:
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())
        data.num_nodes = int(torch.max(data.edge_index)) + 1
    max_degree = max_degree + 4   # edit this!!!!!!!!!!!!!!!!!!!!!!!!!1
  else:
    degs = []
    for data in dataset:
      degs += [degree(data.edge_index[0], dtype=torch.long)]
      data.num_nodes = int(torch.max(data.edge_index)) + 1
    max_degree = max_degree
  if max_degree < 10000:
    # dataset.transform = T.OneHotDegree(max_degree)
    for data in dataset:
      degs = degree(data.edge_index[0], dtype=torch.long)
      data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
  else:
    deg = torch.cat(degs, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()
    for data in dataset:
      degs = degree(data.edge_index[0], dtype=torch.long)
      data.x = ((degs - mean) / std).view(-1, 1)
  return dataset, max_degree