![haldi logo](logo.png)

## haldi

Async-first dependency injection container for Python. It supports transient, scoped, and singleton lifetimes, resolves constructor (or callable) dependencies via type annotations, and can manage async context managers during resolution.

Requires Python 3.10 or newer.

## Features

- Async `resolve()` and `get_executor()` APIs
- Service lifetimes: transient, scoped, singleton
- Multiple registrations per interface (resolve returns a list)
- Ordered union resolution for `typing.Union[...]`, `A | B`, and `Optional[T]`
- Optional lifetime-compatibility warning (singleton depending on scoped)
- Async context manager support for factories

## Installation

```bash
pip install haldi
```

Haldi now targets Python 3.10+.

## Quick start

```python
from haldi import Container

class Repo:
	...

class Service:
	def __init__(self, repo: Repo):
		self.repo = repo

container = Container()
container.add_transient(Repo)
container.add_scoped(Service)

# Resolve inside an async context
service = await container.get(Service)
```

## Service lifetimes

- **Transient**: a new instance every time.
- **Scoped**: one instance per scope. Use `create_scope()` or `scoped()`.
- **Singleton**: one instance for the container lifetime.

```python
container = Container()
container.add_transient(Repo)
container.add_scoped(Service)
container.add_singleton(Config)

scope = container.create_scope()
svc_a = await scope.get(Service)
svc_b = await scope.get(Service)
assert svc_a is svc_b
```

## Multiple implementations

Register multiple providers for the same interface and `resolve()` returns a list.

```python
class Interface:
	...

class ImplA(Interface):
	...

class ImplB(Interface):
	...

container.add_transient(Interface, ImplA)
container.add_transient(Interface, ImplB)

instances = await container.resolve(Interface)
```

## Union annotations

Union-typed dependencies are resolved in declaration order. The first branch that can be resolved wins.

```python
class Cache:
	...

class Repository:
	...

class Service:
	def __init__(self, dep: Cache | Repository | None):
		self.dep = dep
```

If `Cache` is registered, it will be injected. If not, Haldi tries `Repository`. If neither is registered, the final `None` branch is used.

## Factory registration

Factories can be callables (sync or async). Return annotations are required unless using a lambda.

```python
async def build_client() -> Client:
	return Client()

container.add_singleton(Client, build_client)
```

## Scoped context helper

Use `scoped()` to automatically close async context managers created during resolution.

```python
async with container.scoped() as scope:
	service = await scope.get(Service)
```

## Warnings

If a singleton depends on a scoped service, a `UserWarning` is emitted because the scoped instance would be captured for the lifetime of the singleton.

## API overview

- `Container.add_transient(type, concrete_type=None)`
- `Container.add_scoped(type, concrete_type=None)`
- `Container.add_singleton(type, concrete_type=None)`
- `await Container.resolve(type, strict=True)`
- `await Container.get(type)`
- `await Container.try_get(type)`
- `await Container.get_executor(callable, strict=True)`
- `Container.create_scope()`
- `Container.scoped()`

## Development

```bash
pip install -e .
pytest
```
