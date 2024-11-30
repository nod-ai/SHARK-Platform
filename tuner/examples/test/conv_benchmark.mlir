module {
  util.global private @__device_0 = #hal.device.target<"hip", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>]> : !hal.device
  hal.executable private @main_0_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>) {
      hal.executable.export public @main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {}>} {
          %cst = arith.constant 0.000000e+00 : f16
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xi8>>
          %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xi8>>
          %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xi32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xi8>> -> tensor<2x34x34x1280xi8>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xi8>> -> tensor<3x3x1280x1280xi8>
          %5 = tensor.empty() : tensor<2x32x32x1280xi32>
          %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
          %7 = linalg.conv_2d_nhwc_hwcf {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1, 1, 64], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 32, 256, 0, 0, 0]}>} ins(%3, %4 : tensor<2x34x34x1280xi8>, tensor<3x3x1280x1280xi8>) outs(%6 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xi32>>
          return
        }
      }
    }
  }
  util.global private mutable @main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer
  util.initializer {
    %c28190720 = arith.constant 28190720 : index
    %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c28190720}
    util.global.store %buffer, @main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer
    util.return
  }
  util.func public @main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %0 = util.null : !hal.fence
    %c1 = arith.constant 1 : index
    %c10485760 = arith.constant 10485760 : index
    %c17704960 = arith.constant 17704960 : index
    %c14745600 = arith.constant 14745600 : index
    %c2959360 = arith.constant 2959360 : index
    %c0 = arith.constant 0 : index
    %1 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer = util.global.load @main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@main_0_dispatch_0::@rocm_hsaco_fb::@main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@main_0_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main_0_dispatch_0::@rocm_hsaco_fb::@main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32) : index
    scf.for %arg1 = %c0 to %1 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
        (%main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer)[%c0, %c2959360], 
        (%main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer)[%c2959360, %c14745600], 
        (%main_0_dispatch_0_rocm_hsaco_fb_main_0_dispatch_0_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280_i8xi8xi32_buffer : !hal.buffer)[%c17704960, %c10485760]
      ]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%0) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
