.. Copyright 2024 Advanced Micro Devices, Inc.

.. py:module:: _shortfin_default.lib

.. _reference:

API Reference
=============

Array
--------------
.. automodule:: _shortfin_default.lib.array
.. autoclass:: DType
.. autoclass:: storage
    :members:
.. autoclass:: base_array
.. autoclass:: device_array

Local
--------------

.. automodule:: _shortfin_default.lib.local

.. autoclass:: SystemBuilder
.. autoclass:: System
.. autoclass:: Node
.. autoclass:: Device
.. autoclass:: DeviceAffinity
.. autoclass:: Program
.. autoclass:: ProgramFunction
    :members:
.. autoclass:: ProgramModule
.. autoclass:: ProgramInvocation
.. autoclass:: Fiber
.. autoclass:: ScopedDevice
.. autoclass:: Worker
.. autoclass:: Process
.. autoclass:: CompletionEvent
.. autoclass:: Message
.. autoclass:: Queue
.. autoclass:: QueueWriter
.. autoclass:: QueueReader
.. autoclass:: Future
.. autoclass:: VoidFuture
.. autoclass:: MessageFuture


AMD GPU
^^^^^^^
.. automodule:: _shortfin_default.lib.local.amdgpu
.. autoclass:: SystemBuilder
.. autoclass:: AMDGPUDevice

Host
^^^^^^^
.. automodule:: _shortfin_default.lib.local.host
.. autoclass:: CPUSystemBuilder
.. autoclass:: HostCPUDevice
