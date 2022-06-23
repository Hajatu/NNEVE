# `#!python class QOTracker(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

## Introduction

QOTracker class is responsible for collecting the information of the neural network learning process. The information is the value that the class adds to the history of values that occurred during the learning process.

## Instance attributes

`#!python eigenvalue: List[float] = Field(default_factory=list)`

This variable stores the eigenvalue and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python residuum: List[float] = Field(default_factory=list)`

This variable stores the residuum values and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python drive_loss: List[float] = Field(default_factory=list)`

This variable stores the drive loss values and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python function_loss: List[float] = Field(default_factory=list)`

This variable stores the function loss values and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python eigenvalue_loss: List[float] = Field(default_factory=list)`

This variable stores the loss of eigenvalue and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python c: List[float] = Field(default_factory=list)`

This variable stores the C values and automatically while constructing a QOTracker object, this attribute is set to an empty list.
`#!python total_loss: List[float] = Field(default_factory=list)`

This variable stores the values of total loss and automatically while constructing a QOTracker object, this attribute is set to an empty list.

## Instance methods

### `#!python def push_stats(self, ...) -> None`

Called after each learning epoch of the network. Adds new values to the history
of learning process metrics values.

### `#!python def get_trace(self, index: int) -> str:`

Called after tracing the informations after each learning generation of the network. Generates a trace message containting traceback information and displays it as a progress bar.

### `#!python def plot(self,*solution_y: Sequence[float], solution_x: Sequence[float]) -> Figure:`

Called after plotting the graphs after receiving values of learning process of the network. Generates graphs for each variable showing the learning progress.

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
