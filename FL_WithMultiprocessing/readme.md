**USE THIS WHEN YOU HAVE MULTIPLE GPUS OR MULTIPLE CPU CORES**


Then we can achieve pure parallelism while training clients

when we are using google colab free GPUs, we're equipped with only one GPU, so we cannot achieve parallelism in this case.
It is suggested to use when you are training locally using the system's multiple CPU cores, or else you have Multiple GPUs locally.


Even if we use **flower** package, parallel training does not happen by default in Flower simulations because of the way Flower is set up, not because of Colab's single GPU - although the single GPU does limit true parallelism.

By default, Flower's start_simulation() is Sequential
Flower's default simulation is synchronous.
Even if you define num_clients=2, it calls the client_fn() one after another, trains on client 1, then client 2 — sequentially, even on CPU.
