
sources = $(wildcard **/*.java)

objects = $(patsubst %.java,apple_lib/%.class,$(sources))

class_path = ..

apple_lib.jar: $(objects)
	@echo Creating jar file...
	@jar cf apple_lib.jar apple_lib/

clean:
	@rm -rf apple_lib/

apple_lib/%.class: %.java
	@echo Compiling $@
	@javac -d . -cp $(class_path) $<

