# rt

Raytracing [tutorial](http://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf) followed in Rust and Python instead of C++

## Experiments

### 2023-09-28 12:00

```(python)
scene = (
        SceneBuilder()
        .add_sphere(0, 0, -1, 0.5, Materials.METAL.value)
        .add_sphere(0, -100.5, -1, 100, Materials.METAL.value)
        .create()
    )
    camera = Camera(scene, jitter_passes=64)
```

➜  numpy_vector git:(master) ✗ time pypy3.10 main.py
pypy3.10 main.py  599.20s user 4.07s system 98% cpu 10:10.23 total

➜  numpy_vector git:(master) ✗ time python3.11 main.py
python3.11 main.py  127.55s user 0.87s system 100% cpu 2:07.62 total

➜  plain_objects git:(master) ✗ time pypy3.10 main.py
pypy3.10 main.py  6.76s user 0.07s system 98% cpu 6.952 total

➜  plain_objects git:(master) ✗ time python3.11 main.py
python3.11 main.py  75.07s user 0.11s system 99% cpu 1:15.52 total

➜  rt git:(master) time ./target/debug/rt
./target/debug/rt  13.37s user 0.06s system 99% cpu 13.510 total

➜  rt git:(master) ✗ time ./target/release/rt
./target/release/rt  0.57s user 0.01s system 96% cpu 0.599 total
