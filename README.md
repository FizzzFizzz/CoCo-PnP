# CoCo-PnP

## How to test?
> 1. Create environment according to [KAIR](https://github.com/cszn/KAIR), or [DPIR](https://github.com/cszn/DPIR). That is, `pip install -r requirement.txt1 or something like this.
> 2. `python CoCo_ADMM_Poisson_deblur_color.py`
> 3. Check the results in the folder `images/`, and check the PSNR curve entitiled something like `PSNR_level0.18_lamb500.png` in the main folder.

## What does the code mean?
> Take `CoCo_ADMM_Poisson_deblur_color.py` for an example. We are mainly using the `plot_psnr(...)` function. Lines 126-187 is the main CoCo-ADMM code.
> You can write your own codes accordingly. You may need to rewrite the lines 126-187 to realize a new algorithm.
