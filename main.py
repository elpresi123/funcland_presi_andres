# main.py
import re
import sys

# ----------------------------
# Token
# ----------------------------
class Token:
    def __init__(self, type_, value, pos):
        self.type = type_
        self.value = value
        self.pos = pos
    def __repr__(self):
        return f"Token({self.type},{self.value})"


def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]

      
        if c.isspace():
            i += 1
            continue

        
        m = re.match(r'\d+(\.\d+)?', text[i:])
        if m:
            tokens.append(Token("NUMBER", m.group(0), i))
            i += len(m.group(0))
            continue

     
        m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', text[i:])
        if m:
            val = m.group(0)
            if val == "func":
                tokens.append(Token("FUNC", val, i))
            elif val == "print":
                tokens.append(Token("PRINT", val, i))
            else:
                tokens.append(Token("IDENT", val, i))
            i += len(val)
            continue

        if c in '+-*/^(),;=':
            tmap = {
                '+':'PLUS', '-':'MINUS', '*':'MUL', '/':'DIV', '^':'POW',
                '(':'LPAREN', ')':'RPAREN', ',':'COMMA', ';':'SEMI', '=':'EQUAL'
            }
            tokens.append(Token(tmap[c], c, i))
            i += 1
            continue

        raise SyntaxError(f"Error léxico: símbolo desconocido '{c}' en la posición {i}")

    return tokens


class Number:
    def __init__(self, value):
        self.value = float(value) if '.' in str(value) else int(value)
class Var:
    def __init__(self, name):
        self.name = name
class BinaryOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
class Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args
class FuncDef:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body
class PrintStmt:
    def __init__(self, call):
        self.call = call
class Program:
    def __init__(self, funcs, prints):
        self.funcs = funcs
        self.prints = prints

# ----------------------------
# Parser (descendente recursivo)
# ----------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        if self.i < len(self.tokens): return self.tokens[self.i]
        return None

    def consume(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, typ, msg=None):
        t = self.peek()
        if t is None or t.type != typ:
            raise SyntaxError(msg or f"Se esperaba {typ} pero vino {t}")
        return self.consume()

    def parse_program(self):
        funcs = []
        prints = []
        # primero definiciones de funciones (0 o más)
        while self.peek() and self.peek().type == "FUNC":
            funcs.append(self.parse_funcdef())
        # luego prints (0 o más)
        while self.peek() and self.peek().type == "PRINT":
            prints.append(self.parse_print())
        if self.peek() is not None:
            raise SyntaxError(f"Token inesperado al final: {self.peek()}")
        return Program(funcs, prints)

    def parse_funcdef(self):
        self.expect("FUNC")
        name = self.expect("IDENT").value
        self.expect("LPAREN")
        params = []
        if self.peek() and self.peek().type == "IDENT":
            params.append(self.expect("IDENT").value)
            while self.peek() and self.peek().type == "COMMA":
                self.consume()
                params.append(self.expect("IDENT").value)
        self.expect("RPAREN")
        self.expect("EQUAL")
        body = self.parse_expr()
        self.expect("SEMI")
        return FuncDef(name, params, body)

    def parse_print(self):
        self.expect("PRINT")
        call = self.parse_call()
        self.expect("SEMI")
        return PrintStmt(call)

    def parse_call(self):
        name = self.expect("IDENT").value
        self.expect("LPAREN")
        args = []
        # args pueden ser vacíos
        if self.peek() and self.peek().type not in ("RPAREN",):
            args.append(self.parse_expr())
            while self.peek() and self.peek().type == "COMMA":
                self.consume()
                args.append(self.parse_expr())
        self.expect("RPAREN")
        return Call(name, args)

    # expresiones con precedencia
    def parse_expr(self):
        node = self.parse_term()
        while self.peek() and self.peek().type in ("PLUS", "MINUS"):
            op = self.consume().value
            right = self.parse_term()
            node = BinaryOp(op, node, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.peek() and self.peek().type in ("MUL", "DIV"):
            op = self.consume().value
            right = self.parse_factor()
            node = BinaryOp(op, node, right)
        return node

    # '^' es derecho-asociativo: a ^ b ^ c = a ^ (b ^ c)
    def parse_factor(self):
        node = self.parse_primary()
        if self.peek() and self.peek().type == "POW":
            self.consume()
            right = self.parse_factor()  # recursivo para asociatividad derecha
            node = BinaryOp('^', node, right)
        return node

    def parse_primary(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("Expresión incompleta")
        if t.type == "NUMBER":
            self.consume()
            return Number(t.value)
        if t.type == "IDENT":
            # si sigue '(', es llamada, sino variable
            if self.i + 1 < len(self.tokens) and self.tokens[self.i+1].type == "LPAREN":
                return self.parse_call()
            else:
                self.consume()
                return Var(t.value)
        if t.type == "LPAREN":
            self.consume()
            node = self.parse_expr()
            self.expect("RPAREN")
            return node
        raise SyntaxError(f"Token inesperado en expresión: {t}")

# ----------------------------
# Evaluador / Intérprete
# ----------------------------
class SemanticError(Exception):
    pass

class Interpreter:
    def __init__(self, program):
        # guardar definiciones: nombre -> FuncDef
        self.funcs = {}
        for f in program.funcs:
            if f.name in self.funcs:
                raise SemanticError(f"Función duplicada {f.name}")
            self.funcs[f.name] = f
        self.program = program

    def run(self):
        results = []
        for p in self.program.prints:
            try:
                val = self.eval_call(p.call)
                # imprimir como entero si es int exacto
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                print(val)
                results.append(val)
            except SemanticError as e:
                print(f"Error: {e}")
                results.append(None)
        return results

    def eval_call(self, call_node):
        if call_node.name not in self.funcs:
            raise SemanticError(f"función no definida '{call_node.name}'")
        fdef = self.funcs[call_node.name]
        if len(call_node.args) != len(fdef.params):
            raise SemanticError(f"número incorrecto de parámetros en {call_node.name} (esperado {len(fdef.params)}, recibido {len(call_node.args)})")
        # evaluar argumentos (pueden contener llamadas anidadas)
        argvals = [self.eval_expr(a, {}) for a in call_node.args]
        # construir entorno local para la función
        env = dict(zip(fdef.params, argvals))
        return self.eval_expr(fdef.body, env)

    def eval_expr(self, node, env):
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Var):
            if node.name in env:
                return env[node.name]
            else:
                raise SemanticError(f"variable '{node.name}' no definida en ese contexto")
        if isinstance(node, BinaryOp):
            l = self.eval_expr(node.left, env)
            r = self.eval_expr(node.right, env)
            # detectar división por cero
            if node.op == '/':
                if abs(r) < 1e-12:
                    raise SemanticError("división por cero")
                return l / r
            if node.op == '+': return l + r
            if node.op == '-': return l - r
            if node.op == '*': return l * r
            if node.op == '^': return l ** r
            # seguridad
            raise SemanticError(f"operador desconocido {node.op}")
        if isinstance(node, Call):
            # evaluar llamada anidada
            return self.eval_call(node)
        raise SemanticError("Nodo de expresión desconocido")

def main():
    try:
        with open("codigo.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("No se encontró 'codigo.txt' en la carpeta. Crea el archivo y vuelve a ejecutar.")
        return

    try:
        tokens = tokenize(text)
    except SyntaxError as e:
        print(e)
        return

    try:
        parser = Parser(tokens)
        program = parser.parse_program()
    except SyntaxError as e:
        print("Error sintáctico:", e)
        return

    try:
        interp = Interpreter(program)
        interp.run()
    except SemanticError as e:
        print("Error semántico:", e)
        return

if __name__ == "__main__":
    main()
