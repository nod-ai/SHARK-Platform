
Parameter Loading in SHARK-Platform LLM Server
==============================================

Overview
--------
The SHARK-Platform LLM Server uses a flexible parameter loading system to manage large model weights and other resources. Parameters can be loaded from various file formats, including IRPA (IREE Parameter Archive), GGUF, and safetensors. This document explains how parameters are loaded and used in the server.

Quick Start for Users
---------------------
To provide parameters to the SHARK-Platform LLM Server, use the ``--parameters`` command-line argument when starting the server:

.. code-block:: console

   python shortfin/python/shortfin_apps/llm/server.py \
     --parameters path/to/your/params.irpa \
     --parameters path/to/more/params.gguf \
     ...

Multiple parameter files can be specified.

How it works behind the scenes
----------------------------------
Parameter loading is primarily handled by the ``GenerateService`` class, located in ``shortfin/python/shortfin_apps/llm/components/service.py``.

1. Creating StaticProgramPrameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``load_inference_parameters`` method in ``GenerateService`` is responsible for loading parameters:

.. code-block:: python

   def load_inference_parameters(
       self, *paths: Path, parameter_scope: str, format: str = ""
   ):
       p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
       for path in paths:
           logging.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
           p.load(path, format=format)
       self.inference_parameters.append(p)

This method creates a ``StaticProgramParameters`` object for each parameter scope and loads parameters from the provided paths.

2. StaticProgramParameters wraps IREE parameter loading
^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``StaticProgramParameters`` class, defined in ``shortfin/src/shortfin/local/program.h``, wraps the lower-level IREE parameter loading functionality:

.. code-block:: cpp

   class SHORTFIN_API StaticProgramParameters : public BaseProgramParameters {
   public:
     StaticProgramParameters(
         System &system, std::string_view parameter_scope,
         iree_host_size_t max_concurrent_operations =
             IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS);

     struct LoadOptions {
       std::string format;
       bool readable = true;
       bool writable = false;
       bool mmap = true;
     };
     void Load(std::filesystem::path file_path, LoadOptions options);
     void Load(std::filesystem::path file_path) { Load(file_path, LoadOptions()); }

   private:
     iree_allocator_t host_allocator_;
     iree::io_parameter_index_ptr index_;
   };

3. StaticProgramParameters are provided to the inference program
^^^^^^^^^^^^^^^^^^^^^^^^
Loaded parameters are integrated into the inference process when starting the service:

.. code-block:: python

   def start(self):
       self.inference_program = sf.Program(
           modules=[
               sf.ProgramModule.parameter_provider(
                   self.sysman.ls, *self.inference_parameters
               )
           ]
           + self.inference_modules,
           fiber=self.main_fiber,
           trace_execution=False
       )

This creates a ``ProgramModule`` that provides the loaded parameters to the inference modules.

4. Parameter Scopes and Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters are identified by a scope and a unique key within that scope, not by specific file paths. This allows for flexible organization and loading of parameters from different sources.

5. Supported File Formats
^^^^^^^^^^^^^^^^^^^^^^^^^
The server supports multiple parameter file formats:

- IRPA (IREE Parameter Archive): IREE's optimized format for deployment
- GGUF: Used by the GGML project and related ecosystems
- safetensors: Used by the Hugging Face community

6. Parameter File Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^
IREE provides several utilities for working with parameter files:

- ``iree-create-parameters``: Creates IRPA files
- ``iree-convert-parameters``: Converts between supported formats
- ``iree-dump-parameters``: Inspects parameter files

These tools can be used to prepare and manage parameter files for use with the SHARK-Platform LLM Server.

Developer Notes
---------------
- The parameter loading system is designed to be extensible. New file formats can be added by implementing appropriate parsers in the IREE runtime.
- The ``StaticProgramParameters`` class uses IREE's parameter loading APIs, which handle the low-level details of reading and managing parameter data.
- For optimal performance, consider using the IRPA format, which is designed for efficient loading and alignment.
- When developing new models or integrating existing ones, ensure that the parameter scopes and keys match the expectations of the inference modules.

IREE Integration
----------------
The SHARK-Platform LLM Server leverages IREE (IR Execution Environment) for running inference modules. IREE's parameter loading system is used under the hood, providing efficient and flexible parameter management across different devices and deployment scenarios.

For IREE developers, the key integration points are:

- The use of ``iree_io_parameter_provider_t`` for parameter loading
- The creation of parameter modules using ``iree_io_parameters_module_create``
- The integration of loaded parameters into the IREE VM context
