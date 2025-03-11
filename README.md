# Secure Cohesive Subgraph Mining on Multilayer Graphs

`SMPCore` uses [mpyc](https://github.com/lschoe/mpyc) to implement secure multi-party computation.

python version: 3.10

Download dependencies:

```
pip install -r requirements.txt
```

change to the project root directory:

```
cd smpcore
```

## Running the Program in Stand-alone Mode:

### Method 1:

Run the following commands in $m$ terminals respectively, where $m$ is the number of data providers (number of layers in the multilayer graph):

```bash
python code/smpcore.py -M[] -d[] -l[] -v[] -s[] -W[] -I0
```

```bash
python code/smpcore.py -M[] -d[] -l[] -v[] -s[] -W[] -I1
```

...

```bash
python code/smpcore.py -M[] -d[] -l[] -v[] -s[] -W[] -Im-1
```

The values in `[]` are options that must be specified before program execution:

• `-M`: number of providers.

• `-d`: dataset number, datasets 1--4 are `homo`, `Sacchere`,`Sanremo`, `Slashdot` respectively.

• `-l`: value of parameter $\lambda$.

• `-v`: algorithm, algorithms 1--3 are `SMPCore`, `SMPCore-BP`, `SMPCore-AP` respectively.

• `-s`: value of parameter $1/\theta$.

• `-W`: number of threads.

> The parameters of these commands except for -I should be exactly the same.

For example, run the following three commands in three terminals, and the three providers will execute the `SMPCore-AP` algorithm on the `Sanremo` dataset with $\lambda=2, \theta=0.002$, and the number of threads is 4.

```bash
python code/smpcore.py -M3 -d3 -l2 -v3 -s500 -W4 -I0
python code/smpcore.py -M3 -d3 -l2 -v3 -s500 -W4 -I1
python code/smpcore.py -M3 -d3 -l2 -v3 -s500 -W4 -I2
```

### Method 2: (Recommended)

```bash
python ~/smpcore/code/run.py -c'python -u ~/smpcore/code/smpcore.py -M[] -d[] -l[] -v[] -s[] -W[] -I0'
```

For example, run the following command, and the three providers will execute the `SMPCore-AP` algorithm on the `Sanremo` dataset with $\lambda=2, \theta=0.002$, and the number of threads is 4.

```bash
python ~/smpcore/code/run.py -c'python -u ~/smpcore/code/smpcore.py -M3 -d3 -l2 -v3 -s500 -W4 -I0'
```

## Running Commands in Distributed Mode:

Similar to Method 1 in stand-alone mode, run in m terminals respectively:

```bash
python code/smpcore.py -P[ip_1] -P[ip_2] ... -P[ip_m] -M[] -d[] -l[] -v[] -s[] -W[] -I0
```

```bash
python code/smpcore.py -P[ip_1] -P[ip_2] ... -P[ip_m] -M[] -d[] -l[] -v[] -s[] -W[] -I2
```

...

```bash
python code/smpcore.py -P[ip_1] -P[ip_2] -P[ip_3]...-P[ip_m] -M[] -d[] -l[] -v[] -s[] -W[] -Im
```

• -P: IP addresses of different data providers, the order must not be changed

> The parameters of these commands except for -I should be exactly the same

For example, the following three commands on 3 hosts with IP addresses 192.168.31.201, 192.168.31.202, 192.168.31.203 respectively will execute the `SMPCore-AP` algorithm on the `Sanremo` dataset with $\lambda=2, \theta=0.002$, and the number of threads is 4.

```bash
python code/smpcore.py -P 192.168.31.201 -P 192.168.31.202 -P 192.168.31.203 -M3 -d3 -l2 -v3 -s500 -W4 -I0
```

```bash
python code/smpcore.py -P 192.168.31.201 -P 192.168.31.202 -P 192.168.31.203 -M3 -d3 -l2 -v3 -s500 -W4 -I1
```

...

```bash
python code/smpcore.py -P 192.168.31.201 -P 192.168.31.202 -P 192.168.31.203 -M3 -d3 -l2 -v3 -s500 -W4 -I2
```