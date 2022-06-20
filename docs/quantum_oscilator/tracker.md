# `#!python class QOTracker(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

## Introduction

## Instance attributes

## Instance methods

### `#!python def push_stats(self, ...) -> None`

Called after each learning epoch of the network. Adds new values to the history
of learning process metrics values.

#### Parameters

Each parameter will be added to the corresponding class attribute (see names).

| name              | type             |
| ----------------- | ---------------- |
| `total_loss`      | `#!python float` |
| `eigenvalue`      | `#!python float` |
| `residuum`        | `#!python float` |
| `function_loss`   | `#!python float` |
| `eigenvalue_loss` | `#!python float` |
| `drive_loss`      | `#!python float` |
| `c`               | `#!python float` |

#### Returns

| type   | description                    |
| ------ | ------------------------------ |
| Figure | Figure used for drawing plots. |
