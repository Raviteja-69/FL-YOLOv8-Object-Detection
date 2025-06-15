# Federated Learning with Parallel Client Training (Multiprocessing)

This folder contains the code for a Federated Learning (FL) setup that leverages **multiprocessing to enable parallel client training**.

## When to Use This Setup (Achieving True Parallelism)

This approach is designed to achieve **true simultaneous parallel training of multiple clients**, significantly speeding up federated learning rounds. It is most effective when you have:

* **Multiple GPUs:** If you're training locally on a system equipped with more than one GPU. Each client can potentially train on a separate GPU.
* **Multiple CPU Cores:** If you're training locally on a system with several CPU cores. Data loading and model computations for different clients can be distributed across these cores.
* **Distributed Environment:** When clients are running on genuinely separate machines or dedicated compute resources.

## Limitations and What to Expect

### Google Colab Free GPU Tier

If you are using **Google Colab's free GPU tier**, please note the following:

* Colab's free tier typically provides **only one GPU**.
* Even though this code is structured for multiprocessing, you **will NOT achieve true parallel client training** on a single GPU.
* The clients will still process their training rounds **sequentially** on that one GPU. The multiprocessing might manage client execution, but the core GPU computation will be bottlenecked by the single available GPU.

### Understanding Flower's Default Simulation Behavior

Even when using a federated learning framework like **Flower**, it's important to understand its default simulation behavior:

* Flower's `start_simulation()` function, by default, is **synchronous**.
* This means it will call the `client_fn()` for `Client 1`, wait for it to complete, then call the `client_fn()` for `Client 2`, and so on.
* This **sequential execution** happens by default in simulation mode, **even if you have multiple CPU cores or GPUs available locally**, unless you specifically configure Flower for true parallel client execution (e.g., by using a custom client manager or a distributed setup for actual client processes).

**In summary:** While this `FL_Multiprocessing` setup is built for parallelism, its effectiveness depends entirely on the underlying hardware resources (multiple GPUs/CPUs) or a truly distributed system. On a single GPU environment like Google Colab's free tier, clients will still execute one after another.
