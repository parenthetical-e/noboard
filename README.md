# noboard
Sometimes you don't need tensorboard, or any progress board at all. You just need to log some data.

The `noboard.SummaryWriter` mocks a part of the familiar `tensorboard.SummaryWriter` interface, but logs data to csv files or other simple common formats. 

NOTE: This is an early version. A work in progress. Especially tensor -> scalar conversion.

# motivation
- Tensorboard saves its data to a hidden away json file. You could parse that, for much the same effect. But the format is to me opaque. This library is a simpler answer that promises its data format is stable, and clear. 

- `noboard` matches the `tensorboard.SummaryWriter` interface, as best it can, to make it easy to change between them.

# dependencies
- python >= 3.6
- numpy

# install
```bash
git clone git@github.com:parenthetical-e/noboard.git
cd noboard
pip install .
```
    
# usage
```python
from noboard.csv import SummaryWriter
writer = SummaryWriter(log_dir=None) # creates runs/
for n in range(100):
    writer.add_scalar("reward", reward, n)
```
