# lakematrix：用 Python + JAX 重写 LAKE 热力学核心（全矩阵化，CPU 先行）

本文件是交给执行者的工程说明。执行者没有之前的对话上下文，所有需要的事实都写在这里。
遇到与本文不符的代码事实，以代码为准，并在 `lakematrix/NOTES.md` 记录差异。

英文版见 `LAKEMATRIX-PLAN.en.md`，内容相同；两者有出入时以英文版为准。

---

## 0. 背景与目标

- LAKE 3.2 是一维（垂直）湖泊模型，Fortran，约 2.3 万行有效代码。
  每个时间步对**一个湖**调用一次 `SUBROUTINE LAKE`（`core/source/model/lake.f90:169-3212`）。
  多湖时，驱动程序循环逐个湖调用（`core/source/driver/driver.f90:765` 附近）。
- 目标：新写一个 **以"湖"为数组维度** 的模型。输入是 (湖, 时间)，所有状态是 (湖, …) 数组，
  每个物理过程一次处理全部湖，输出直接是 (湖, 时间[, 深度]) NetCDF。不存在逐湖循环。
- 已定的决策：
  1. Python + JAX，**float64**（`jax.config.update("jax_enable_x64", True)`），先只做 CPU，GPU 以后再说。
  2. 正确性按**容差**判定，参照为现有 Fortran 模型（见 §5）。
  3. 第一版只做**热力学核心**：水温、盐度扩散（淡水）、k-epsilon 湍流、动量、冰、雪、
     土壤（含多土壤柱与侧向热交换）、表面热量/动量通量、短波辐射。
     **不做**：甲烷/氧/CO2/磷/叶绿素、气泡（BUBBLE）、入流/出流、Flake、数据同化、restart。
  4. **独立的新包** `lakematrix`，不依赖 lakebatch 的 Python 代码。
- 全局变量的处理：Fortran 的模块全局变量和 SAVE 局部变量，在新模型里全部变成
  显式 State 中的 (湖, …) 数组，靠广播运算。不需要任何"逐湖换入换出"的机制。

## 1. 工作环境与约定（必须遵守）

- 代码根目录：`/Users/mac/文档/研究生/LAKE/`（Mac 与 Linux VM 共享同一目录）。
  - `lakebatch/`：现有项目（Fortran 核心 `core/`、Python 包 `src/lakebatch/`、各种工具）。
    本计划对它**只读**，唯一例外是 §2 的收尾和 §4 的参照构建放在 `lakebatch/scratch/` 下。
  - `LAKE3.2/`：原版模型目录，`LAKE3.2/` 下有 testlake 示例（`make_cases.py` 从这里取 setup）。
  - 新包放在 `/Users/mac/文档/研究生/LAKE/lakematrix/`（新建，独立 git 仓库）。
- 机器：Linux VM 是 arm64，5 核，无 GPU，有网络（`pip install jax` 可用）。用户的 Mac 也会运行。
  - VM 的 Fortran 构建产物在 `lakebatch/core/build-linux/`（生产）和 `core/build-check-linux/`（check）；
    **绝不碰** Mac 的 `core/build/` 构建产物和 `core/lake.out`。
  - VM 的 Python 虚拟环境：`lakebatch/.venv-linux/`（lakebatch 用）。lakematrix 自建 `lakematrix/.venv-linux/`。
- Fortran 构建：
  - `lakebatch/core/build.sh`：生产构建（`-O3`，含 FMA 融合）。
  - `CHECK=1 lakebatch/core/build.sh`：check 构建（`-O3 -ffp-contract=off`），**参照数据一律用 check 构建**，
    因为 FMA 融合会让末位随代码布局变化。
- 文件操作约定：**不用 `rm -rf`**。每次试验写到新目录（例如 `.../ref/v2/`），不要清空旧目录。
  确需删除时，先说明删除什么。
- git：lakematrix 新仓库，每个里程碑通过验收后提交一次。作者用
  `GIT_AUTHOR_NAME="陈少春" GIT_AUTHOR_EMAIL="116076400+ThywQuake@users.noreply.github.com"`
  （以及同值的 `GIT_COMMITTER_*`）环境变量，不改 git config。
- 测试：`pytest`。长测试（60/425 天）标记 `@pytest.mark.slow`，默认跳过，`pytest -m slow` 运行。

## 2. 第 0 步：收尾 lakebatch 分支（小）

lakebatch 当前分支 `rewrite/phased-step`，最新提交 `11790ea`（阶段 1b）。工作区有一处未提交修改：
`vector/rewrite/README.md` 末尾追加的 "BUBBLE's share of run time" 一节。

1. 提交这处修改（提交信息说明：BUBBLE 占 26–29%，阶段 2–3 不做）。
2. 在 `lakebatch/HANDOVER.md` 和 `lakebatch/vector/STATUS.md` 顶部各加一段：
   阶段 1（多湖同进程、逐湖与单跑逐位一致）已完成；阶段 2–3 取消；转向 `../lakematrix`（JAX 全矩阵重写）。
   阶段 1 的多湖 Fortran 保留为参照工具。
3. 提交。

## 3. 要翻译的 Fortran 范围

### 3.1 一步的结构（`lake.f90` 中 `SUBROUTINE LAKE`）

| 段落 | 行号（约） | 内容 |
|---|---|---|
| 首次调用初始化 | 530–810 | 强迫赋值、`INIT_VAR`（`init_var_mod.f90:7-500`，仅首次）、`kor`、辐射消光系数（653–659） |
| 读入当前状态 | 811 | `UPDATE_CURRENT_TIMESTEP`（`lake_modules.f90:1138-1335`）：从 `*_2d` 数组取出本湖状态 |
| 网格与情况判定 | 855–912 | `BATHYMETRY`（`bathym_mod.f90:7-382`）、`time/nstep` 推进、**layer_case 判定（881–912，见 3.2）** |
| case 1 无冰 | 916–1238 | `BATHYMETRY`、`DENSITY_W`、`MIXED_LAYER_CALC`、`RadWater%RAD_UPDATE`（1000）、`MOMENTUM_SOLVER`（1049）、`T_SOLVER`（1070）+ `SOILFORLAKE`（1073）、`S_DIFF`（1091）、（生物地化 1095–1153，**跳过**）、`DENSITY_W`、`K_EPSILON`（1162）、`TURB_SCALES`（1216） |
| case 2 冰+水 | 1239–1897 | 冰/雪辐射、`T_SOLVER`（1463）、`SNOWTEMP`（1491）、`MOMENTUM_SOLVER`（1549）、`K_EPSILON`（1559）等；冰的生长融化、雪 |
| case 3 只有冰（冻透） | 1898–2086 | `T_SOLVER`（1960）、`SNOWTEMP`（1969）等 |
| case 4 | 2087– | **不可运行**（原代码 STOP），不翻译 |
| 层厚修正 | 约 2153–2270 | 水/冰/雪层厚度变化、最小厚度处理 |
| 写回状态 | 2575 | `UPDATE_NEXT_TIMESTEP`（`lake_modules.f90:1340-1534`） |
| 输出 | 2640–3150 | 不翻译（新包自己的输出，见 M6） |

执行者的第一项工作（M0.3）是把上表细化为 `lakematrix/docs/step_map.md`：
case 1–3 中每个调用的顺序、条件（`if`）、读写了哪些状态变量。这是后续所有翻译的路线图。

### 3.2 layer_case 判定（原文，`lake.f90:881-912`）

```
layer_case = 1
if (h1 >  0 .and. l1 >  0) layer_case = 2
if (h1 == 0 .and. l1 >  0) layer_case = 3
if (h1 == 0 .and. l1 == 0) layer_case = 4
case 1 且表层水达到冰点（WATER_FREEZE_MELT(Tw1(1), …, +1)）且水够厚 → case 2
case 3 且冰底可融（WATER_FREEZE_MELT(Ti1(Mice+1), …, -1)）→ case 2
case 3 且冰面可融且无雪 → 生成薄水层（h1 = min_water_thick, Tw1 = 冰点 + T_phase_threshold, …）→ case 1
```
JAX 中先按湖算出 `layer_case`（(湖,) 整数）；第三条规则会改写状态，用 `jnp.where` 按掩码改写。

### 3.3 热力学核心子程序（文件:行号）

数值/通用：`PROGONKA` numerics_mod.f90:243-273、`MATRIXPROGONKA` numerics_mod.f90:108-172、
`DZETA`/`DZETAI`/`VARMEAN` dzeta_mod.f90、`DIFFUSION`（diffusion_lake_mod）、`VERTADV_UPDATE`/`MUSCL_FLUXES`/`FLUX_LIMITER`（vertadv）、
`LININTERPOL`（grid_operations_lake）、`IND_STAB_FACT_DB`、`STEP`、`KRON`、`INVERSMATRIX`、`MATRIXMULT*`、`MATRIXSUM`、`VECTORSUM`（numerics）。

物理：
- 温度：`T_SOLVER` t_solver_mod.f90:15-216、`T_DIFF` 219-656、`DIFF_COEF` 662-937、`SURF_CHAR` 940-1034
- 表面：`HEATBALANCE` heatbalsurf_mod.f90:8-72、`SENSLATMOM_FLUXES` 167-384、`RADBALANCE` 388-508、
  `HEATFLUXSURF`、（`TSKIN_SOLVER` 644-748：testlake `skin 0`，不用）；`SfcFlx_*` shared/SfcFlx_mod.F90
  （`momsenlat`、`roughness`、`rhoair` 623-687、`satwvpres` 855-936、`spechum` 949-1012、`lwradwsfc` 1019-1081）
- 湍流：`K_EPSILON` turb_mod.f90:10-1158、`CE_CANUTO`/`CET_CANUTO`/`INTERPOLATE_TO_INTEGER_POINT`（turb_lake_mod）、
  `TURB_SCALES` phys_func.f90:914-1069、`MIXED_LAYER_CALC` 1072-1154、`RICHARDSON_NUMBER`、`TURB_DENS_FLUX`、`DIFFMIN_HS`、`DOMWAVESPEED`
- 动量：`MOMENTUM_SOLVER` momentum_mod.f90:15-1611
- 密度：`DENSITY_W` phys_func.f90:160-201、water_density 模块（`WATER_DENS_TS`、`DDENS_DTEMP`、`DDENS_DSAL`、`SET_DDENS_*`、`SET_DENS_CONSTS` phys_func.f90:60-86），testlake `eos 5`
- 辐射：`RAD_UPDATE` lake_modules.f90:386-403、`RAD_INIT` 406-418、`WATER_ALBEDO`
- 冰雪：`SNOWTEMP` snowtemp.f90:6-97、`SNOW_CALC` snowcalc.f90:5-497、`MELTPNT`/`MELTINGPOINT`/`WATER_FREEZE_MELT`/`WL_MAX`/`SOIL_COND_JOHANSEN`（phys_func）
- 土壤：`SOILFORLAKE` soil_mod.f90:215-709、`COMSOILFORLAKE` 8-212、`LATERHEAT` 1385-1628、`SOILCOLSTEMP`、`TSURFSOILCOL`、`AVERSOILCOL`、`SOIL_COND_HEAT_COEF`
- 其它：`S_DIFF` salinity_mod.f90:14-228、`ZERODIM_MODEL` t_0dim.f90:4-113（每步约 5 次调用，确认其作用后再决定是否需要）、
  `BATHYMETRY` bathym_mod.f90:7-382、`JULIAN_DAY`（time_lake_mod）
- 初始化/常数：`INIT_LAKE` lake.f90:30-159、`INIT_VAR` init_var_mod.f90:7-500、`DEFINE_PHYS_CONSTANTS` lake_modules.f90:195-309、
  `DEFINE_PHYS_PARAMETERS` 119-128、`NUMERIC_PARAMS_SET` 99-104、`TURB_CONST_DERIVED_SET` 903-937、
  `PBLDAT_LAKE` surf_scheme1_mod.f90:6-100、`PBLDAT_INM` surf_scheme_inm_mod.f90:4-185、`ALLOCARRAYS` init.f90:177-483、
  `DEFINE_DRIVING_PARAMS` driving_params_mod.f90:211-476（setup 关键字读取）

这份清单来自一次 testlake 运行的 gprof 调用记录（每步调用次数参考：`HEATBALANCE` 8 次、`DIFF_COEF` 27、`T_DIFF` 13、
`PROGONKA` 32、`SOILFORLAKE` 5、`LATERHEAT` 4、`MOMENTUM_SOLVER`/`K_EPSILON`/`T_SOLVER` 各 1）。
调用次数大于 1 的说明内部有迭代或按土壤柱/介质循环，翻译时必须弄清循环结构。

### 3.4 跨步状态的完整清单（不能漏）

跨时间步的状态 = 下面三部分之和。State 设计（M1）必须覆盖全部，或在 `NOTES.md` 写明为何不需要：
1. `UPDATE_CURRENT_TIMESTEP` / `UPDATE_NEXT_TIMESTEP` 读写的 `*_2d`/`*_3d` 数组（热力学相关）：
   `Tw1 Ti1 Tis1 T Tsoil1 wi1 wl1 Sal1 Sals1 salice porice dens lamw u v E eps h1 l1 hs1 ls1 hx1 hx2 hy1 hy2 hx1t hx2t hy1t hy2t
   hx1ml hx2ml hy1ml hy2ml gradpx_tend gradpy_tend dhw dhw0 dhi dhi0 dls0 dhwfsoil snmelt snowmass fl_sn fl_sn_init Tskin
   roughness cdm velfrict eflux0_kinem Elatent k_turb_T_flux i_maxN itherm itop dz wl time nstep Eseiches`
   （生物地化的 `qwater oxyg DIC DOC POCL POCD phosph carbdi qsoil oxygsoil febul0 tot_ice_meth_bubbles` 不要。）
2. 模块变量与子程序 SAVE 局部变量：lakebatch 阶段 1 已生成完整清单，即
   `lakebatch/core/source/model/lb_point_ctx.f90`（每个 `LBC_MOD_<模块>` 子程序列出该模块跨调用的变量）
   和 `lb_ctx_<routine>.inc`（`LAKE`、`K_EPSILON`、`MOMENTUM_SOLVER`、`T_SOLVER`、`T_DIFF`、`DIFF_COEF`、
   `SURF_CHAR`、`SNOWTEMP`、`SNOW_CALC`、`SOILFORLAKE`、`COMSOILFORLAKE`、`SENSLATMOM_FLUXES` 的 SAVE 局部变量）。
   其中大部分是"每步重算"的工作数组，只有"先读后写"的才是真正跨步状态；执行者逐个判定，结论写进 `docs/state.md`。
3. 首次调用块（`firstcall`）里算的常数：放进 Params/常数，不放 State。

## 4. 参照数据（Fortran）

### 4.1 参照配置

- 基础：`lakebatch/vector/rewrite/make_cases.py` 的 `make_case(dst, forcings, tinteg)` 从 `LAKE3.2` 的 testlake 生成算例
  （`hourly 0`、`npoints`、`dataname 'test'`、`spinup_times 0`）。`_rows()` 读取 testlake 强迫，`_cold(rows)` 把气温降 2 K。
  testlake：2016-08-01 03:00 开始，`dt 20` 秒，强迫逐小时，`M 40`、`Mice 10`、`ns 10`、`nsoilcols 5`、`Turbpar 2`、
  `soilswitch 1`、`eos 5`、`skin 0`、`carbon_model 2`、`ifbubble 1`、`tribheat 3`、`phi 56`、深 22.66 m、`lakeform 2` + 25 层形态表。
  强迫文件 8 列（逗号分隔）：1 气温 K、2 气压 Pa、3 长波下行、4 短波下行、5 u、6 v、7 比湿 kg/kg、8 降水。
- 热力学参照配置（新写 `lakematrix/tools/ref/make_ref_cases.py`，调用上面的 `make_case` 后改 setup）：
  `tribheat 0`（关入流热量），其余不变。
- **先验证生物地化对热力学没有反馈**（M0.4）：生物地化调用（`lake.f90:1095-1153` 及 case 2/3 的对应段）和
  `BUBBLE_BLOCK` 是无条件执行的，没有总开关。在 `lakebatch/scratch/thermo-only/` 复制一份 `core/source`，
  把这些调用注释掉，用 check 标志构建（构建方法见 4.3），跑 425 天，与未改的 check 构建比较
  `water_temp`、`ice_temp`、`soil_temp`、`layers`、`u`、`v`、`epsilon`、`lamw`、`water_dens_layers` 等时间序列：
  - 完全一致 → 参照用原版 check 构建即可，JAX 只需对比热力学输出；
  - 不一致 → 找出反馈通道（可能是消光、密度或土壤），记录在 `NOTES.md`；参照改用注释掉生物地化的构建，
    并把这个构建方法固化进 `tools/ref/`。

### 4.2 参照算例集（`lakematrix/ref/v1/`）

| 名称 | 湖 | 天数 | 用途 |
|---|---|---|---|
| `summer1d` | testlake | 1 | M3 快速对比 |
| `summer60d` | testlake | 60（8 月–9 月） | M3 |
| `full425d` | testlake | 425 | M4（含结冰、融冰、雪） |
| `cold425d` | testlake，气温 −2 K | 425 | M4，与 full425d 同批时同一步处于不同 layer_case |
| `shallow425d` | testlake，`h10` 改为 5 m、形态表按比例缩放 | 425 | 浅湖（更早冻透，可能触发 case 3） |
| `warm425d` | testlake，气温 +3 K、`phi 40` | 425 | 不同纬度/少冰 |

每个算例都是单湖 Fortran 运行（用 check 构建）。批量对比时，JAX 一次跑这些湖组成的批，逐湖与对应单湖参照比较。
若 `shallow425d` 没有出现 case 3，可再降低深度或气温直到出现（目的是覆盖全部三种情况），在 `NOTES.md` 记录最终参数。

### 4.3 子程序级参照：探针构建（probe build）

目的：每个 JAX 子程序单独与 Fortran 比对，不必等整步写完。

- 在 `lakebatch/scratch/probe-build/` 下：复制 `lakebatch/core/source` 为 `source/`，复制 `lakebatch/core/build/CMakeLists.txt`
  为 `build/CMakeLists.txt`，并用 sed 做与 `core/build.sh` 中 CHECK 分支相同的两处替换
  （`RUNTIME_OUTPUT_DIRECTORY "../"` → `"."`；`list(APPEND FLAGS_ "-O3")` → 加 `"-ffp-contract=off"`）。
  构建：`cd build && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_Fortran_FLAGS="-I$PWD/../source/model" . && cmake --build . --target lake.out`。
  （`lakebatch/scratch/bubble-share/` 就是用这种方式构建的，可参考。）
- 新写 Fortran 模块 `source/model/lm_probe.f90`（加入 CMakeLists 的 model 源列表），提供：
  - 通用接口 `LM_PROBE(tag, name, x)`：对 real(8)/integer/logical 的标量与 1–3 维数组，
    当当前步号 `nstep` 在环境变量 `LM_PROBE_STEPS`（逗号分隔）中时，追加写入
    `probe/<nstep>/<tag>.bin`：每个变量一条记录 = 名字（64 字节，空格填充）+ dtype 码（1 字节：f/i/b）
    + 维数（int64）+ 各维上下界（int64）+ 数据（小端，Fortran 列序）。
  - Python 读取器 `lakematrix/tools/ref/probe.py`：`read_probe(path) -> dict[name, np.ndarray]`
    （按 Fortran 列序 reshape，记录下界以便对齐 0 起始/1 起始）。
- 在探针构建的源码里，于每个要比对的子程序调用**前后**插入 `LM_PROBE("<routine>_in"/"<routine>_out", …)`，
  记录该子程序的全部输入（含它读取的模块变量）和全部输出（含它修改的模块变量与 SAVE 变量）。
  插入用脚本完成（`tools/ref/make_probe_build.py`，对源码做文本补丁），不要手改，以便可重复。
- 取样步：夏季（case 1）若干步、结冰当步、冬季冰下（case 2）若干步、融冰当步、
  有雪的步、（若有）case 3 的步。具体步号在 M0.4 从 `full425d`/`shallow425d` 的 `layers` 输出中挑选，写进 `tools/ref/steps.toml`。
- 注意：Fortran 数组下界常为 0 或 1（例如 `fbbleflx(0:M+1)`、`Tw1(1:M+1)`、`ddz05(0:M)`）。JAX 一律 0 起始，
  在 `docs/state.md` 中为每个数组写明对应关系。

## 5. 验收标准（容差）

标准分两类：**诊断**标准很严，用于定位翻译错误；**验收**标准是相对 Fortran 参照的有物理意义的等价性。诊断标准超标时，若原因已在 `NOTES.md` 记录为迭代噪声，不单独判定里程碑失败；验收标准超标则必须修复。

| 层级 | 比较对象 | 标准 |
|---|---|---|
| 子程序，确定性模块 | 探针前后状态，单步 | 每个输出数组：`max|a-b| <= 1e-10 * max(1, max|b|)`。同公式同顺序时应接近逐位 |
| 子程序，迭代模块（`T_SOLVER`、`K_EPSILON`、动量等） | 探针前后状态，单步 | 迭代次数与收敛残差必须一致；输出在该求解器噪声底之内（见 `NOTES.md`，目前 `T_SOLVER` 温度取 1e-3 K） |
| 整步，非迭代字段 | 探针整步前后（`UPDATE_CURRENT_TIMESTEP` 后 → `UPDATE_NEXT_TIMESTEP` 前） | 相对误差 1e-8；`layer_case`、各 flag、所有迭代次数必须一致 |
| 整步，迭代求解字段（`Tw1`、`Tsoil1`/`Tsoil2`、`Tis1`、`Tsurf`、`E`、`eps` 等） | 同上 | 在 `NOTES.md` 记录的对应求解器噪声底之内；迭代次数必须一致 |
| 1 天 | `summer1d` 输出 | 水温 ≤ 1e-3 K；u、v ≤ 1e-6 m/s |
| 60 天 | `summer60d` | 逐日逐点水温 ≤ 0.5 K；时间平均剖面偏差 ≤ 0.1 K；剖面 RMSE ≤ 0.15 K |
| 425 天 | `full425d` 等 | 月平均水温 ≤ 1.5 K；全期平均 ≤ 0.5 K；结冰日、融冰日相差 ≤ 7 天；冰厚、雪厚 ≤ 0.3 m。不比较逐点轨迹（模型对初值/末位差异呈混沌放大，见下） |
| 批次一致性 | 同一批 N 个湖 vs 每湖单独（N=1）跑 JAX | 每个湖每个输出 ≤ 1e-12（确认湖之间无串扰；在 CPU 上应逐位一致） |

原“整步 1e-10”和“1 天 1e-6 K”低于 Fortran 求解器的固有噪声：`T_SOLVER` 的停止判据是 `|Bal| <= 1 W/m²`，约相当于 1e-4 K 的表面温度。两个独立实现即使公式正确、只是 Chorde 迭代次数不同，也无法满足这两条。

原“60 天 0.01 K”和“425 天 0.05 K”同样低于 Fortran 自身的敏感性：把强迫中一个值改动 1 ULP，60 天剖面最大差 0.27 K（时间平均 0.012 K），425 天剖面最大差 3.44 K（月平均 1.17 K，融冰日相差 5 天，冰厚差 0.22 m）。模型在季节尺度上是混沌的，两条同源二进制的运行都无法满足逐点阈值。修订后的长期标准比较平均季节循环与冰情物候；翻译正确性由单步探针和 ≤1 天标准保证。复核与数据见 `NOTES.md`；确定性模块仍保持 1e-10。
比较脚本：`lakematrix/tools/compare.py REF_CASE_DIR OUT.nc LAKE_INDEX`。读 Fortran 时间序列
（`results/test/time_series/<var>  1  1.dat`：前 5 列为 年 月 日 时 积分天数，之后为 深度/值 成对；
文件名中含空格，需要加引号）和 `layers  1  1.dat`（列说明在文件头 `Col. N - …` 行），
与 JAX 输出 NetCDF 按时间、深度对齐后给出每变量最大误差与是否通过。

## 6. 新包设计

```
lakematrix/
  pyproject.toml          # 依赖：jax[cpu]>=0.4.30, numpy, netCDF4, xarray；可选 test: pytest
  README.md  NOTES.md
  docs/step_map.md docs/state.md
  src/lakematrix/
    __init__.py           # 启用 x64
    constants.py          # DEFINE_PHYS_CONSTANTS/PARAMETERS、TURB_CONST、NUMERIC_PARAMS 的数值（frozen dataclass 或 dict）
    config.py             # Params：每湖参数，NamedTuple/pytree，叶子是 (L,) 或 (L, k) 数组
    setup_io.py           # 读 LAKE 的 setup/driver .dat 关键字（单湖），组装 Params；支持多个湖拼成批
    grid.py               # 无量纲网格（DZETA/DZETAI）、ddz/ddz05/ddzi…，形态表插值（BATHYMETRY 的静态部分）
    state.py              # State pytree（见下）+ init_state(params, grid)（INIT_VAR 的热力学部分）
    forcing.py            # 强迫：(L, T) 数组；从 NetCDF 或 LAKE 强迫 .dat 读入；逐步插值方式与 Fortran 一致
    physics/
      tridiag.py          # 批量追赶法：输入 (L, n) 的 a,b,c,d；lax.scan 沿 n
      blocktridiag.py     # MATRIXPROGONKA（动量方程的 2x2 块三对角）
      density.py  radiation.py  surface.py  tsolver.py  soil.py  snow.py  ice.py
      turb.py  momentum.py  salinity.py  layers.py  bathym.py
    step.py               # step(state, forcing_t, params, grid) -> state：全部湖一起
    run.py                # run(...)：jit(step)；lax.scan 按块（如 1 天 = 4320 步）推进，块间取输出
    output.py             # (lake, time[, depth]) NetCDF
    cli.py                # lakematrix run setup_list.txt|input.nc out.nc
  tools/ref/              # make_ref_cases.py, make_probe_build.py, probe.py, steps.toml
  tools/compare.py
  tests/
```

### 6.1 数据约定

- 维度顺序：湖在最前。L = 湖数。水层 `(L, M+1)`，冰层 `(L, Mice+1)`，土壤 `(L, nsoilcols, ns)`，标量 `(L,)`。
- 第一版要求同一批所有湖的 `M`、`Mice`、`ns`、`nsoilcols` 相同（Fortran 本身是无量纲坐标，
  深度不同只改变 `h1`），`setup_io` 遇到不同值时报错。
- State、Params、Grid 都是 pytree（`typing.NamedTuple` 或 `flax.struct` 风格的 dataclass + `jax.tree_util.register_dataclass`）。
  不在 State 里放 Python 标量或字符串。
- 函数全部是纯函数，不用全局可变变量。Fortran 的 SAVE 变量要么进 State，要么证明可以每步重算。

### 6.2 分支

- `layer_case`：每步按 §3.2 计算 (L,) 整数。**第一版**：对全部湖执行 case 1、case 2、case 3 三套更新，
  用 `jnp.where(layer_case == k, …)` 按湖选择结果（对 State 的每个叶子做 tree_map 选择）。
  这保证正确且没有数据依赖的控制流。M5 再优化（统一公式或按情况分组）。
- 子程序内部的 `if`：一律改写为 `jnp.where`，两边都要能安全计算（避免除零、log 负数产生 NaN 后被 where 选中的
  NaN 梯度/传播问题：先用 `jnp.where` 修正输入再计算）。
- 迭代（`K_EPSILON` 内部迭代、表面温度迭代、动量等）：`lax.while_loop`，条件为"存在未收敛的湖且未达最大次数"，
  每轮只更新未收敛的湖（`jnp.where(done, old, new)`），收敛判据和最大次数与 Fortran 完全相同。
  这样每个湖得到与单独迭代相同的结果。
- 沿垂直方向的递推（追赶法、逐层积分）：`lax.scan` 沿层维，湖维保持向量化。

### 6.3 输出

- NetCDF：维度 `lake`、`time`、`depth`（固定输出深度，取 testlake 的 `ngrid_out` 26 个深度 0–24 m，可配置）、`ice_depth`、`soil_depth`。
- 变量（第一版）：`water_temp(lake,time,depth)`、`ice_temp`、`soil_temp`、`u`、`v`、`tke`、`epsilon`、
  `h_water`、`h_ice`、`h_snow`、`h_deepice`、`t_surface`、`sensible`、`latent`、`layer_case`。
- 输出频率默认每小时（与 Fortran `time_series` + `dt_out 1.` 一致），用于逐时对比。

## 7. 里程碑与任务

每个里程碑结束：测试全过 → 更新 `NOTES.md`（做了什么、误差实测值、遗留问题）→ 提交。

### M0 准备（无 JAX 物理代码）
1. 第 0 步（§2）收尾 lakebatch。
2. 新建 `lakematrix/` 仓库、`pyproject.toml`、`.venv-linux`，`pip install -e .[test]`；`python -c "import jax; jax.config.update('jax_enable_x64', True); print(jax.devices())"` 能运行。
3. 写 `docs/step_map.md`（§3.1 细化）：对 case 1/2/3 列出每个调用、条件、读写变量。
4. 验证生物地化无反馈（§4.1），生成参照算例集（§4.2），挑选探针步（§4.3），完成探针构建与 `probe.py`。
   检查：`read_probe` 读出的 `Tw1` 与同一步 Fortran `water_temp` 输出一致（插值到输出深度后）。
5. 写 `docs/state.md`（§3.4 判定结果 + 下界对应关系）。

### M1 骨架
- `constants.py`、`config.py`、`setup_io.py`、`grid.py`、`state.py`（`init_state`）、`forcing.py`。
- 测试：对 testlake，`init_state` 与探针记录的第 1 步 `UPDATE_CURRENT_TIMESTEP` 之后的状态一致（1e-12）；
  网格量（`ddz` 等）与探针一致；强迫插值后的逐步值与探针中 `LAKE` 入口的强迫标量一致。
- 测试：两份 setup（testlake 与 cold）拼批后，各湖字段等于各自单独构建的结果。

### M2 基础计算
- `tridiag.py`、`blocktridiag.py`、`density.py`、`radiation.py`、`surface.py`（SfcFlx_* + SENSLATMOM_FLUXES + RADBALANCE + HEATBALANCE + HEATFLUXSURF）、`bathym.py`。
- 测试：每个函数用探针输入算，与探针输出比（1e-10）；tridiag 另用随机矩阵与 `numpy.linalg.solve` 比。
- 测试：批次一致性（N 个不同湖 vs 各自 N=1）。

### M3 无冰情况（case 1）
- `tsolver.py`（T_SOLVER/T_DIFF/DIFF_COEF/SURF_CHAR 中 case 1 用到的分支）、`soil.py`（SOILFORLAKE、LATERHEAT、多土壤柱）、
  `salinity.py`、`momentum.py`、`turb.py`（K_EPSILON + TURB_SCALES + MIXED_LAYER_CALC）、`layers.py`（case 1 的层厚变化）、
  `step.py` 中 case 1 路径、`run.py`、`output.py`、`tools/compare.py`。
- 测试：每个子程序对夏季探针步（1e-10）；整步（1e-10）；`summer1d`、`summer60d` 整体（§5）。
- 复核长期容差（§5 末段）。

### M4 冰雪（case 2、3）与情况切换
- `ice.py`、`snow.py`（SNOWTEMP/SNOW_CALC）、`tsolver.py` 的冰/雪/深冰分支、case 2/3 路径、§3.2 的切换规则（含 case 3→1 的状态改写）。
- 测试：冬季/结冰/融冰/有雪/（case 3）探针步（1e-10）；`full425d`、`cold425d`、`shallow425d`、`warm425d` 整体（§5）。
- 测试：四个湖同批跑 425 天，逐湖与各自参照比较，并与 JAX N=1 逐位一致。

### M5 性能（CPU）
- `run.py`：`jax.jit` 整个块（`lax.scan` 若干步），避免每步 Python 调度；
  `jax.block_until_ready` 计时，首次编译时间单独记录。
- 基准：testlake 复制 L = 1、8、64、512、4096 个湖（可对强迫加小扰动），60 天。
  指标：湖·步/秒。对照：Fortran check 构建与生产构建，单进程单湖的湖·步/秒，以及 5 核同时跑 5 个进程的总吞吐量。
- 剖析（`jax.profiler` 或 `XLA_FLAGS=--xla_dump_to`）找瓶颈；优先处理"三种情况全算"的浪费和 while_loop 的开销。
- 结果写进 `README.md` 表格。

### M6 输出与命令行
- `cli.py`：`lakematrix run <setup 列表或 input.nc> <out.nc> [--days N] [--chunk-steps K]`。
- 输入：①多个 LAKE setup/driver 目录列表（参照算例直接可用）；②沿用 lakebatch schema v1 的 (lake, time) NetCDF
  （格式见 `lakebatch/src/lakebatch/inputs.py` 文件头文档字符串；只实现热力学所需关键字）。
- 端到端测试：用 CLI 跑 4 湖 425 天，`tools/compare.py` 全部通过。

## 8. 风险与处理

- **范围大**（约 1 万行 Fortran）：严格按子程序探针逐个比对，任何子程序过不了 1e-10 不进入下一个。
- **隐藏状态**：SAVE 变量与模块变量容易漏；§3.4 的清单是完整的（来自 lakebatch 阶段 1 的生成器），以它为准。
- **NaN 与 where**：被掩掉的分支也会计算，必须保证输入合法（见 6.2）。
- **while_loop 在 CPU 上的开销**：M5 实测；必要时改为固定次数迭代（需证明结果在容差内）。
- **CPU 提速有限**：约 34% 时间在 exp/log 等数学函数；M5 的实测数据用于决定是否尽快做 GPU 版本。
- **浮点顺序**：JAX/XLA 可能重排求和，子程序级可能达不到逐位一致，因此标准是 1e-10 而非逐位。
  若某处超出，先检查是否公式或顺序翻译错误，再考虑是重排造成。
