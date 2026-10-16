"""Allow `python -m mineru_benchmark.eval.t3_deep_analysis` to invoke main()."""

from mineru_eval.t3_deep_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
