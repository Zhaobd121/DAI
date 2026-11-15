DAI_FLAGS = -std=c++17 -O3 -march=native -Wall
DAI_SRC = main.cpp $(wildcard *.hpp)
VARIANTS = $(basename $(notdir $(wildcard ablation/*.patch)))

all: dai $(VARIANTS)

.PHONY: all clean $(VARIANTS)

dai: $(DAI_SRC)
	$(CXX) $(DAI_FLAGS) main.cpp -o dai

$(VARIANTS): %: ablation/%/dai

ablation/%/dai: ablation/%.patch $(DAI_SRC)
	rm -rf ablation/$*
	mkdir -p ablation/$*
	cp *.hpp main.cpp ablation/$*
	patch -d ablation/$* -p1 < $<
	cd ablation/$* && $(CXX) $(DAI_FLAGS) -DVARIANT='"$*"' main.cpp -o dai

clean:
	rm -rf dai $(addprefix ablation/, $(VARIANTS))
