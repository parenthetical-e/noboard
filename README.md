# noboard
Sometimes you don't need tensorboard, or any progress board at all. You just need to log some data.

The `noboard.SummaryWriter` mocks a part of the familiar `tensorboard.SummaryWriter` interface, but logs data to csv files or other simple common formats. 

NOTE: This is an early version. A work in progress.

# dependencies
- python >= 3.6

# install
```bash
git clone git@github.com:parenthetical-e/noboard.git
cd noboard
pip install -e .
```
    
# usage
```python
from noboard import SummaryWriter
writer = SummaryWriter(log_dir=None) # creates runs/
for n in range(100):
    writer.add_scalar("reward", reward, n)
```
