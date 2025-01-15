
sources = $(wildcard **/*.java)

objects = $(patsubst %.java,apple_lib/%.class,$(sources))

class_path = ..

args = -d . -cp $(class_path) -Xlint:unchecked

apple_lib.jar: $(objects)
	@echo Creating jar file...
	@jar cf apple_lib.jar apple_lib/

clean:
	@rm -rf apple_lib/

apple_lib/%.class: %.java
	@echo Compiling $@
	@javac $(args) $<

