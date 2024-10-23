import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';

function Header() {

    return (
        <Navbar expand="lg" className="bg-dark navbar-dark">
            <Container>
                <Navbar.Brand href="/">Utilities</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="me-auto">
                    <Nav.Link href="/saltosLinea">Borrar saltos de linea</Nav.Link>
                    <Nav.Link href="/modelos">Crear modelos de ML</Nav.Link>
                </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    )
}

export default Header;