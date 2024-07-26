# UATReinforce
A Reinforcement Learning Framework for Urban Airspace Tradable Permit Model

## Compilation (during debugging)

```sh
git submodule update --init --recursive
make -H. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=on -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Compilation (for release)

```sh
git submodule update --init --recursive
make -H. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=on -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/uatsim -h
```
