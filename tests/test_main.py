# tests

def test_hello(capfd):
    from hello_world_package import hello
    hello.say_hello()
    out, err = capfd.readouterr()
    assert out == "Hello World!\n"
