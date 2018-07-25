Grokking PyTorch
================

[PyTorch](https://pytorch.org/) is a fast and flexible deep learning framework that allows automatic differentiation through dynamic neural networks (i.e., networks that utilise dynamic control flow like if statements and while loops).

Neural networks are a subclass of *computation graphs*. Computation graphs receive input data, and data is routed to and possibly transformed by nodes which perform processing on the data. In deep learning, the neurons (nodes) in neural networks typically transform data with parameters and differentiable functions, such that the parameters can be optimised to minimise a loss via gradient descent. More broadly, the functions can be stochastic, and the structure of the graph can be dynamic. So while neural networks are well suited for [dataflow programming](https://en.wikipedia.org/wiki/Dataflow_programming), PyTorch's API is more centred around the typical [imperative programming](https://en.wikipedia.org/wiki/Imperative_programming) paradigm to make it easier to read and reason about control flow, and hence complex programs.

The rest of this document, based on the [official MNIST example](https://github.com/pytorch/examples/tree/master/mnist), is about *grokking* PyTorch, and should only be looked at after the [official beginner tutorials](https://pytorch.org/tutorials/). For readability, the code is presented in chunks interspersed with comments, and hence not separated into different functions/files as it would normally be for clean, modular code.

Imports
-------

```py
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

These are pretty standard imports, with the exception of the `torchvision` modules that are used for computer vision problems in particular.

Setup
-----

```py
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
  torch.cuda.manual_seed(args.seed)
```

`argparse` is a standard way of dealing with command-line arguments in Python.

A good way to write device-agnostic code (benefitting from GPU acceleration when available but falling back to CPU when not) is to pick and save the appropriate `torch.device`, which can be used to determine where tensors should be stored. See the [official docs](https://pytorch.org/docs/master/notes/cuda.html#device-agnostic-code) for more tips on device-agnostic code. The PyTorch way is to put device placement under the control of the user, which may seem a nuisance for simple examples, but makes it much easier to work out where tensors are - which is useful for a) debugging and b) making efficient use of devices manually.

For repeatable experiments, it is necessary to set random seeds for anything that uses random number generation (including `random` or `numpy` if those are used too). Note that cuDNN uses nondeterministic algorithms, and it can be disabled using `torch.backends.cudnn.enabled = False`.

Data
----

```py
train_data = datasets.MNIST('data', train=True, download=True,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.MNIST('data', train=False, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True)
```

`torchvision.transforms` contains lots of handy transformations for single images, such as cropping and normalisation.

`DataLoader` contains many options, but beyond `batch_size` and `shuffle`, `num_workers` and `pin_memory` are worth knowing for efficiency. `num_workers` > 0 uses subprocesses to asynchronously load data, rather than making the main process block on this. `pin_memory` uses [pinned RAM](https://pytorch.org/docs/master/notes/cuda.html#use-pinned-memory-buffers) to speed up RAM to GPU transfers (and does nothing for CPU-only code).

Model
-----

```py
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

model = Net().to(device)
optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if args.resume:
  model.load_state_dict('model.pth')
  optimiser.load_state_dict('optimiser.pth')
```

Network initialisation typically includes member variables, layers which contain trainable parameters, and maybe separate trainable parameters and non-trainable buffers. The forward pass then uses these in conjunction with functions from `F` that are purely functional (don't contain parameters). Some people prefer to have completely functional networks (e.g., keeping parameters separately and using `F.conv2d` instead of `nn.Conv2d`) or networks completely made of layers (e.g., `nn.ReLU` instead of `F.relu`).

`.to(device)` is a convenient way of sending the device parameters (and buffers) to GPU if `device` is set to GPU, doing nothing otherwise (when `device` is set to CPU). It's important to transfer the network parameters to the appropriate device before passing them to the optimiser, otherwise the optimiser will not be keeping track of the parameters properly!

Both neural networks (`nn.Module`) and optimisers have the ability to save and load their internal state, and `.load_state_dict(state_dict)` is the recommended way to do so - you'll want to reload the state of both to resume training from previously saved state dictionaries. Saving the entire object can be [error prone](https://pytorch.org/docs/stable/notes/serialization.html).

Some points of note not shown here are that the forward pass can make use of control flow (e.g., a member variable or even the data itself can determine the execution of an if statement. It is also perfectly valid to `print` tensors in the middle, making debugging much easier. Finally, the forward pass can make use of multiple arguments. A short snippet (not tied to any sensible idea) to illustrate this is below:

```py
def forward(self, x, hx, drop=False):
  hx2 = self.rnn(x, hx)
  print(hx.mean().item(), hx.var().item())
  if hx.max.item() > 10 or self.can_drop and drop:
    return hx
  else:
    return hx2
```

Training
--------

```py
losses = []

net.train()


  losses.append(loss.item())
```

Network modules are by default set to training mode - which impacts the way some modules work, most noticeably dropout and batch normalisation. It's best to set this manually anyway with `.train()`, which propagates the training flag down all children modules.

It's important not to accidentally store the graph! See the [official docs](https://pytorch.org/docs/stable/notes/autograd.html) for more on autograd.
Talk about `.detach()` for RL?

Talk about accumulating grads


Testing
-------

```py
net.eval()

with torch.no_grad():
```

In response to earlier, networks should explicitly be set to evaluation mode using `.eval()`.

Because of the way autograd works, the computation graph is kept during evaluation; `with torch.no_grad()` stops this happening.
