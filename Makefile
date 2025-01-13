
util_obj = out/apple_lib/utility/*.class
env_obj = out/apple_lib/environment/*.class
network_obj = out/apple_lib/network/*.class out/apple_lib/network/layer/*.class
lp_obj = out/apple_lib/lp/*.class

objects = $(util_obj) $(env_obj) $(network_obj) $(lp_obj)

class_path = ..

apple_lib.jar: $(objects)
	@echo Creating jar file...
	@$(MAKE) -C out
	@mv out/apple_lib.jar .

out/apple_lib/%.class: %.java
	@echo Compiling $@
	@javac -d out/ -cp $(class_path) $<

