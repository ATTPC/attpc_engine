# Frequently Asked Questions

## How do I install attpc_engine
See [Setup and Installation](user_guide/setup.md)

## How do I get started using attpc_engine
See [Getting Started](user_guide/getting_started.md)

## How do I contribute to attpc_engine
See [For Developers](for_devs.md)

## How do I know if attpc_engine can be used with my dataset
In general attpc_engine is flexible, and can generate data for many different types of AT-TPC experiments. However, the detector system side of attpc_engine is rather fixed at the moment. Serious modifications to the AT-TPC geometry and behavior may make certain datasets incompatible, as well as attpc_engine is not designed to simulate the behavior of additional detector components. But the kinematics sampler should be valid for most types of reactions with AT-TPC.

## How can I look at the kinematics data without detector effects applied
See [Converting to Dataframes](./user_guide/kinematics/index.md#converting-to-dataframes)

## How to make attpc_engine output data into a different format
See [Format the point cloud](./user_guide/detector/index.md#format-the-point-cloud)

## How do I update my version of attpc_engine

With your virtual environment active you can use any of

```bash
pip install attpc_engine --upgrade
pip install attpc_engine -U
```

or to install a specific version use

```bash
pip install attpc_engine==x.x.x
```
and replace `x.x.x` with the specific version number to install.

